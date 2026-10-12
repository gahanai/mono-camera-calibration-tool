#pragma once

// Radar clutter filtering: the 7-stage ordered pipeline from radcam_calibplan.md §6, cheapest to
// most expensive, with mandatory per-stage funnel counts so a capture that yields zero
// correspondences can be traced to the exact stage that ate it.
//
// Stage order (do not reorder -- each stage's position is deliberate per spec):
//   0. Background subtraction (strongest, cheapest, do first)
//   1. Spatial gate (iteration-dependent radius around the predicted radar-frame position)
//   2. Doppler gate (reflector is static -> near-zero Doppler)
//   3. Range-adaptive RCS gate (two-sided -- rejects both too-dim AND implausibly-bright returns)
//   4. Ground-plane plausibility (catches multipath ghosts)
//   5. Best-in-gate by RCS, per frame (defeats antenna sidelobes -- NOT "closest to prediction")
//   6. Dwell persistence + median aggregation across frames

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "bev/radarcam/can_decoder.hpp"
#include "bev/radarcam/capture_io.hpp"
#include "bev/radarcam/types.hpp"

namespace bev {
namespace radarcam {

struct ClutterFilterResult {
    bool accepted = false;
    std::string reject_reason;
    cv::Point3d q_radar{0.0, 0.0, 0.0};  // median accepted position, radar frame
    double dwell_spread_m = 0.0;         // QC metric: max distance of any accepted frame from the median
    double detection_rcs_dbsm = 0.0;     // representative (median) RCS of the accepted detections
    ClutterFunnel funnel{};
};

struct ClutterFilterParams {
    double gate_radius_m = 2.0;  // current iteration's spatial gate radius -- wide first pass, tight later
    double doppler_threshold_mps = 0.5;
    double sigma_trihedral_dbsm = 20.0;
    double r_ref_m = 1.0;
    double rcs_margin_db = 10.0;
    double background_match_radius_m = 0.3;
    double radar_height_above_ground_m = 0.5;
    double persistence_frac = 0.8;
    double spread_max_m = 0.10;

    // 2D radar (e.g. DRWIG LRR Pro, drwig_decoder.hpp): detections all lie in the radar's z=0 plane
    // while q_hat keeps the reflector's true height, so every distance compares x/y only.
    bool planar = false;
    // false when the radar reports no RCS: stage 3 passes everything through, and stage 5 picks the
    // detection closest to q_hat instead of the brightest (there is no brightness to rank by).
    bool has_rcs = true;
};

namespace detail {
inline double euclidean_distance(const cv::Point3d& a, const cv::Point3d& b) { return cv::norm(a - b); }

inline double gate_distance(const cv::Point3d& a, const cv::Point3d& b, bool planar) {
    return planar ? std::hypot(a.x - b.x, a.y - b.y) : euclidean_distance(a, b);
}
}  // namespace detail

// q_hat = R_est^-1 * (p_camera - t_est): the reflector's expected radar-frame position given the
// current camera-side prediction and extrinsic estimate (§6 stage 1's q_hat formula).
inline cv::Point3d predict_radar_frame_position(
    const cv::Point3d& p_camera, const cv::Matx33d& R_est, const cv::Vec3d& t_est) {
    const cv::Vec3d p(p_camera.x, p_camera.y, p_camera.z);
    const cv::Vec3d q = R_est.inv() * (p - t_est);
    return cv::Point3d(q[0], q[1], q[2]);
}

// Runs the full 7-stage pipeline for a single capture's dwell against the predicted radar-frame
// reflector position `q_hat` (see predict_radar_frame_position). `background` may be empty if no
// background scan is available -- stage 0 becomes a no-op in that case. Every background detection
// (whether from a genuine per-capture scan or a single session-wide fallback scan -- that fallback
// POLICY choice is the caller's responsibility, not this function's) is pooled into one flat set
// and matched against by proximity, per spec's "match detections... by proximity" wording.
inline ClutterFilterResult run_clutter_pipeline(const RadarDwell& dwell, const RadarDwell& background,
    bool has_background, const cv::Point3d& q_hat, const ClutterFilterParams& params) {
    ClutterFilterResult result;
    result.funnel = {StageFunnelCount{"background_subtraction", 0, 0}, StageFunnelCount{"spatial_gate", 0, 0},
        StageFunnelCount{"doppler_gate", 0, 0}, StageFunnelCount{"rcs_gate", 0, 0},
        StageFunnelCount{"ground_plausibility", 0, 0}, StageFunnelCount{"best_in_gate", 0, 0},
        StageFunnelCount{"persistence_aggregation", 0, 0}};

    // ---- Stage 0: background subtraction ----
    std::vector<RadarDetection> bg_pool;
    if (has_background) {
        for (const auto& bg_frame : background.frames) {
            bg_pool.insert(bg_pool.end(), bg_frame.begin(), bg_frame.end());
        }
    }
    std::vector<std::vector<RadarDetection>> after_bg(dwell.frames.size());
    {
        int entering = 0, surviving = 0;
        for (size_t fi = 0; fi < dwell.frames.size(); ++fi) {
            for (const auto& d : dwell.frames[fi]) {
                ++entering;
                bool matched = false;
                for (const auto& bg_d : bg_pool) {
                    if (detail::gate_distance(d.position, bg_d.position, params.planar) <=
                        params.background_match_radius_m) {
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    after_bg[fi].push_back(d);
                    ++surviving;
                }
            }
        }
        result.funnel[0].n_entering = entering;
        result.funnel[0].n_surviving = surviving;
    }

    // ---- Stage 1: spatial gate ----
    std::vector<std::vector<RadarDetection>> after_spatial(after_bg.size());
    {
        int entering = 0, surviving = 0;
        for (size_t fi = 0; fi < after_bg.size(); ++fi) {
            for (const auto& d : after_bg[fi]) {
                ++entering;
                if (detail::gate_distance(d.position, q_hat, params.planar) <= params.gate_radius_m) {
                    after_spatial[fi].push_back(d);
                    ++surviving;
                }
            }
        }
        result.funnel[1].n_entering = entering;
        result.funnel[1].n_surviving = surviving;
    }

    // ---- Stage 2: Doppler gate ----
    std::vector<std::vector<RadarDetection>> after_doppler(after_spatial.size());
    {
        int entering = 0, surviving = 0;
        for (size_t fi = 0; fi < after_spatial.size(); ++fi) {
            for (const auto& d : after_spatial[fi]) {
                ++entering;
                if (std::abs(d.vel_mps) <= params.doppler_threshold_mps) {
                    after_doppler[fi].push_back(d);
                    ++surviving;
                }
            }
        }
        result.funnel[2].n_entering = entering;
        result.funnel[2].n_surviving = surviving;
    }

    // ---- Stage 3: range-adaptive RCS gate (two-sided) ----
    std::vector<std::vector<RadarDetection>> after_rcs(after_doppler.size());
    {
        const double r_hat = cv::norm(q_hat);
        const double expected_dbsm = (r_hat > 1e-6)
            ? params.sigma_trihedral_dbsm - 40.0 * std::log10(r_hat / params.r_ref_m)
            : params.sigma_trihedral_dbsm;
        int entering = 0, surviving = 0;
        for (size_t fi = 0; fi < after_doppler.size(); ++fi) {
            for (const auto& d : after_doppler[fi]) {
                ++entering;
                if (!params.has_rcs || std::abs(d.rcs_dbsm - expected_dbsm) <= params.rcs_margin_db) {
                    after_rcs[fi].push_back(d);
                    ++surviving;
                }
            }
        }
        result.funnel[3].n_entering = entering;
        result.funnel[3].n_surviving = surviving;
    }

    // ---- Stage 4: ground-plane plausibility ----
    std::vector<std::vector<RadarDetection>> after_ground(after_rcs.size());
    {
        int entering = 0, surviving = 0;
        for (size_t fi = 0; fi < after_rcs.size(); ++fi) {
            for (const auto& d : after_rcs[fi]) {
                ++entering;
                if (d.position.z >= -params.radar_height_above_ground_m) {
                    after_ground[fi].push_back(d);
                    ++surviving;
                }
            }
        }
        result.funnel[4].n_entering = entering;
        result.funnel[4].n_surviving = surviving;
    }

    // ---- Stage 5: best-in-gate by RCS, per frame ----
    std::vector<RadarDetection> per_frame_best;  // at most one entry per frame
    {
        int entering = 0, surviving = 0;
        for (const auto& frame : after_ground) {
            entering += static_cast<int>(frame.size());
            if (frame.empty()) continue;
            const auto best = params.has_rcs
                ? std::max_element(frame.begin(), frame.end(),
                      [](const RadarDetection& a, const RadarDetection& b) { return a.rcs_dbsm < b.rcs_dbsm; })
                : std::min_element(frame.begin(), frame.end(), [&](const RadarDetection& a, const RadarDetection& b) {
                      return detail::gate_distance(a.position, q_hat, params.planar) <
                             detail::gate_distance(b.position, q_hat, params.planar);
                  });
            per_frame_best.push_back(*best);
            ++surviving;
        }
        result.funnel[5].n_entering = entering;
        result.funnel[5].n_surviving = surviving;
    }

    // ---- Stage 6: dwell persistence + median aggregation ----
    {
        const int entering = static_cast<int>(per_frame_best.size());
        result.funnel[6].n_entering = entering;

        const int n_frames_total = static_cast<int>(dwell.frames.size());
        const double persistence = (n_frames_total > 0)
            ? static_cast<double>(per_frame_best.size()) / static_cast<double>(n_frames_total)
            : 0.0;

        if (!(persistence > params.persistence_frac)) {
            result.funnel[6].n_surviving = 0;
            result.accepted = false;
            result.reject_reason = "Dwell persistence (" + std::to_string(persistence) +
                ") does not exceed required fraction (" + std::to_string(params.persistence_frac) + ").";
            return result;
        }

        std::vector<double> xs, ys, zs, rcs;
        xs.reserve(per_frame_best.size());
        ys.reserve(per_frame_best.size());
        zs.reserve(per_frame_best.size());
        rcs.reserve(per_frame_best.size());
        for (const auto& d : per_frame_best) {
            xs.push_back(d.position.x);
            ys.push_back(d.position.y);
            zs.push_back(d.position.z);
            rcs.push_back(d.rcs_dbsm);
        }
        auto median_of = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const size_t n = v.size();
            return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
        };
        const cv::Point3d median_pos(median_of(xs), median_of(ys), median_of(zs));

        double max_spread = 0.0;
        for (const auto& d : per_frame_best) {
            max_spread = std::max(max_spread, detail::gate_distance(d.position, median_pos, params.planar));
        }

        if (!(max_spread < params.spread_max_m)) {
            result.funnel[6].n_surviving = 0;
            result.accepted = false;
            result.reject_reason = "Positional spread across dwell (" + std::to_string(max_spread) +
                " m) exceeds spread_max_m (" + std::to_string(params.spread_max_m) + " m).";
            return result;
        }

        result.funnel[6].n_surviving = entering;
        result.accepted = true;
        result.q_radar = median_pos;
        result.dwell_spread_m = max_spread;
        result.detection_rcs_dbsm = median_of(rcs);
        return result;
    }
}

}  // namespace radarcam
}  // namespace bev
