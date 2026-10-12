#pragma once

// Iterative gate tightening (radcam_calibplan §7.3): gate wide on the coarse extrinsic -> solve
// (Kabsch+RANSAC) -> re-gate tight with the improved extrinsic -> re-solve, for a few passes.
// Includes the mandatory guardrail: if the inlier count drops sharply as the gate tightens, the
// gate has become over-constrained and is now selecting detections that merely agree with the
// current estimate, quietly biasing the solve toward what was already believed -- a warning is
// recorded on that iteration's record rather than silently trusting the tighter-but-smaller
// consensus.
//
// §11 note: `planar` selects the reduced-DOF path (yaw + in-plane translation, planar_solve.hpp) for
// a 2D radar; pitch/roll/height then stay fixed at coarse_R/coarse_t for every iteration. Callers
// must also set params.planar so the clutter filter gates in the radar plane.

#include "bev/radarcam/board_pnp.hpp"
#include "bev/radarcam/clutter_filter.hpp"
#include "bev/radarcam/kabsch.hpp"
#include "bev/radarcam/planar_solve.hpp"
#include "bev/radarcam/ransac.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace bev {
namespace radarcam {

// One capture's input to the iterate loop. Board pose is retained per the spec's own
// Correspondence-dataclass comment ("retained for §8 refinement") even though this module's
// closed-form solve never touches it -- §9 nonlinear refinement needs it later.
struct CaptureInput {
    cv::Point3d p_camera{0.0, 0.0, 0.0};  // predicted reflector position, camera frame (from PnP)
    BoardPose board_pose;
    RadarDwell dwell;
    RadarDwell background;
    bool has_background = false;
};

struct IterationRecord {
    int iteration = 0;
    double gate_radius_m = 0.0;
    int n_candidates = 0;  // captures that survived clutter filtering and were offered to RANSAC
    int n_inliers = 0;
    bool guardrail_triggered = false;
    std::string guardrail_message;
};

struct GateIterateResult {
    bool success = false;
    RigidTransform transform;
    std::vector<int> final_inlier_indices;         // indices into `captures`
    std::vector<ClutterFilterResult> per_capture;  // clutter-filter outcome per capture, last iteration run
    std::vector<IterationRecord> iterations;
};

// Runs the gate -> solve -> re-gate loop. `gate_radii_m` gives one radius per iteration (spec's
// "wide ~2.0m -> tight ~0.3-0.5m", typically 2-3 entries); `coarse_R`/`coarse_t` seed the first
// iteration's spatial gate. Other ClutterFilterParams fields are held fixed across iterations.
inline GateIterateResult run_gate_iterate(const std::vector<CaptureInput>& captures,
    const std::vector<double>& gate_radii_m, const cv::Matx33d& coarse_R, const cv::Vec3d& coarse_t,
    ClutterFilterParams params, double ransac_inlier_threshold_m, double inlier_drop_warn_frac,
    bool planar = false) {
    GateIterateResult result;
    if (gate_radii_m.empty()) {
        return result;
    }

    cv::Matx33d current_R = coarse_R;
    cv::Vec3d current_t = coarse_t;
    int previous_inliers = -1;

    for (size_t iter = 0; iter < gate_radii_m.size(); ++iter) {
        params.gate_radius_m = gate_radii_m[iter];

        std::vector<ClutterFilterResult> per_capture(captures.size());
        std::vector<cv::Point3d> p_candidates, q_candidates;
        std::vector<int> candidate_capture_indices;

        for (size_t ci = 0; ci < captures.size(); ++ci) {
            const auto& cap = captures[ci];
            const cv::Point3d q_hat = predict_radar_frame_position(cap.p_camera, current_R, current_t);
            per_capture[ci] =
                run_clutter_pipeline(cap.dwell, cap.background, cap.has_background, q_hat, params);
            if (per_capture[ci].accepted) {
                p_candidates.push_back(cap.p_camera);
                q_candidates.push_back(per_capture[ci].q_radar);
                candidate_capture_indices.push_back(static_cast<int>(ci));
            }
        }

        IterationRecord record;
        record.iteration = static_cast<int>(iter);
        record.gate_radius_m = params.gate_radius_m;
        record.n_candidates = static_cast<int>(p_candidates.size());

        if (p_candidates.size() < 3) {
            result.iterations.push_back(record);
            result.per_capture = per_capture;
            return result;  // not enough correspondences to solve this iteration
        }

        const auto ransac_result = planar
            ? solve_ransac_planar(p_candidates, q_candidates, coarse_R, coarse_t, ransac_inlier_threshold_m)
            : solve_ransac(p_candidates, q_candidates, ransac_inlier_threshold_m);
        if (!ransac_result.success) {
            result.iterations.push_back(record);
            result.per_capture = per_capture;
            return result;
        }
        record.n_inliers = static_cast<int>(ransac_result.inlier_indices.size());

        if (previous_inliers > 0) {
            const double drop_frac =
                1.0 - (static_cast<double>(record.n_inliers) / static_cast<double>(previous_inliers));
            if (drop_frac > inlier_drop_warn_frac) {
                record.guardrail_triggered = true;
                record.guardrail_message = "Inlier count dropped by " + std::to_string(drop_frac * 100.0) +
                    "% when tightening the gate to " + std::to_string(params.gate_radius_m) +
                    "m -- the gate may be over-constrained, selecting only detections that already "
                    "agree with the current estimate.";
            }
        }
        previous_inliers = record.n_inliers;
        result.iterations.push_back(record);

        current_R = ransac_result.transform.R;
        current_t = ransac_result.transform.t;

        result.success = true;
        result.transform = ransac_result.transform;
        result.per_capture = per_capture;
        result.final_inlier_indices.clear();
        for (int local_idx : ransac_result.inlier_indices) {
            result.final_inlier_indices.push_back(candidate_capture_indices[local_idx]);
        }
    }

    return result;
}

}  // namespace radarcam
}  // namespace bev
