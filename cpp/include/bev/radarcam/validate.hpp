#pragma once

// Validation (radcam_calibplan §9) -- two independent axes, both required, plus a pre-solve
// diagnostic:
//   §9.1 Held-out residuals: 3D RMS/median/p95 and pixel reprojection error on captures excluded
//        from the solve.
//   §9.2 Ground-plane cross-check: decomposes the EXISTING Homography-tab YAML (via
//        bev::load_homography_yaml, already in calibration_io.hpp -- no new parsing) into a full
//        camera<->ground pose, composes it with the solved radar<->camera transform, and derives
//        the radar's mounting height/pitch/roll for comparison against the tape-measure/
//        inclinometer values. Read-only: this module has no dependency edge back into
//        kabsch.hpp/gate_iterate.hpp/refine_ceres.hpp's inputs, only their outputs -- it must never
//        feed into the solve (coplanar ground-placed points would reintroduce exactly the vertical
//        degeneracy the 4D radar was chosen to avoid).
//   §9.3 Capture-distribution diagnostic (run before solving): SVD of the centered reflector
//        position cloud: warn if the smallest/largest singular-value ratio is below threshold
//        (default 0.15), and report range/azimuth/height spread separately, flagging height
//        specifically since it's "the one operators forget."

#include "bev/calibration_io.hpp"
#include "bev/radarcam/kabsch.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace bev {
namespace radarcam {

// ---------------------------------------------------------------------------------------------
// §9.1 Held-out residuals
// ---------------------------------------------------------------------------------------------

struct HeldoutInput {
    cv::Point3d p_camera{0.0, 0.0, 0.0};
    cv::Point3d q_radar{0.0, 0.0, 0.0};
};

struct HeldoutResidualStats {
    int n_holdout = 0;
    double rms_3d_m = 0.0;
    double median_3d_m = 0.0;
    double p95_3d_m = 0.0;
    double rms_reprojection_px = 0.0;
    double median_reprojection_px = 0.0;
    double p95_reprojection_px = 0.0;
};

inline cv::Point2d project_pinhole(const cv::Matx33d& K, const cv::Point3d& p) {
    // Rectified image, no distortion left to account for -- see the rectify-before-fusion
    // convention shared with board_pnp.hpp/HomographyTab::estimate_camera_pose_pnp.
    const double x = p.x / p.z;
    const double y = p.y / p.z;
    return cv::Point2d(K(0, 0) * x + K(0, 2), K(1, 1) * y + K(1, 2));
}

inline double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    if (values.size() == 1) return values[0];
    const double rank = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) return values[lo];
    const double frac = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

inline HeldoutResidualStats compute_heldout_residuals(
    const std::vector<HeldoutInput>& heldout, const RigidTransform& transform, const cv::Matx33d& camera_matrix) {
    HeldoutResidualStats stats;
    stats.n_holdout = static_cast<int>(heldout.size());
    if (heldout.empty()) return stats;

    std::vector<double> residuals_3d, residuals_px;
    residuals_3d.reserve(heldout.size());
    residuals_px.reserve(heldout.size());

    for (const auto& h : heldout) {
        const cv::Vec3d qv(h.q_radar.x, h.q_radar.y, h.q_radar.z);
        const cv::Vec3d predicted = transform.R * qv + transform.t;
        const cv::Point3d predicted_pt(predicted[0], predicted[1], predicted[2]);
        residuals_3d.push_back(cv::norm(predicted_pt - h.p_camera));

        const cv::Point2d px_camera = project_pinhole(camera_matrix, h.p_camera);
        const cv::Point2d px_radar = project_pinhole(camera_matrix, predicted_pt);
        residuals_px.push_back(cv::norm(px_camera - px_radar));
    }

    double sq_sum_3d = 0.0, sq_sum_px = 0.0;
    for (double r : residuals_3d) sq_sum_3d += r * r;
    for (double r : residuals_px) sq_sum_px += r * r;

    stats.rms_3d_m = std::sqrt(sq_sum_3d / static_cast<double>(residuals_3d.size()));
    stats.median_3d_m = percentile(residuals_3d, 0.5);
    stats.p95_3d_m = percentile(residuals_3d, 0.95);
    stats.rms_reprojection_px = std::sqrt(sq_sum_px / static_cast<double>(residuals_px.size()));
    stats.median_reprojection_px = percentile(residuals_px, 0.5);
    stats.p95_reprojection_px = percentile(residuals_px, 0.95);
    return stats;
}

// ---------------------------------------------------------------------------------------------
// §9.2 Ground-plane cross-check
// ---------------------------------------------------------------------------------------------

struct GroundPlaneCrossCheck {
    bool success = false;
    std::string error;
    double derived_radar_height_m = 0.0;
    double radar_pitch_deg = 0.0;  // tilt of the radar's forward (+x) axis out of the level plane
    double radar_roll_deg = 0.0;   // rotation about the forward axis
};

// Decomposes a WORLD(ground)->PIXEL homography into a full ground->camera rigid pose, via the
// standard H = K*[r1 r2 t] decomposition for a calibrated camera viewing a planar target (columns
// of K^-1*H give r1, r2, t up to a common scale fixed by normalizing r1/r2 to unit length; r3
// completes the rotation via cross product; re-orthogonalized with the same SVD+determinant-
// correction trick as solve_kabsch, since r1/r2 won't be perfectly orthogonal under any noise).
inline bool decompose_ground_homography(const cv::Matx33d& H_world_to_pixel, const cv::Matx33d& camera_matrix,
    cv::Matx33d& R_world_to_camera, cv::Vec3d& t_world_to_camera, std::string& error) {
    const cv::Matx33d M = camera_matrix.inv() * H_world_to_pixel;

    const cv::Vec3d col0(M(0, 0), M(1, 0), M(2, 0));
    const cv::Vec3d col1(M(0, 1), M(1, 1), M(2, 1));
    const cv::Vec3d col2(M(0, 2), M(1, 2), M(2, 2));

    const double norm0 = cv::norm(col0);
    const double norm1 = cv::norm(col1);
    if (norm0 < 1e-9 || norm1 < 1e-9) {
        error = "Degenerate homography (zero-norm column) -- cannot decompose.";
        return false;
    }
    const double lambda = 2.0 / (norm0 + norm1);

    const cv::Vec3d r1 = col0 * lambda;
    const cv::Vec3d r2 = col1 * lambda;
    const cv::Vec3d r3 = r1.cross(r2);

    // Re-orthogonalizing a SINGLE approximately-orthogonal matrix (r1/r2 won't be perfectly
    // orthogonal under noise) is the "nearest orthogonal matrix" problem, whose solution is
    // R = U*V^T from R_raw's own SVD -- NOT the V*U^T formula solve_kabsch uses, which answers a
    // different question (the optimal rotation aligning two separate point sets via their
    // cross-covariance matrix). Mixing the two up gives a wildly wrong rotation despite looking
    // like "the same SVD + determinant trick".
    cv::Matx33d R_raw(r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]);
    cv::Mat U, S, Vt;
    cv::SVD::compute(cv::Mat(R_raw), S, U, Vt, cv::SVD::FULL_UV);
    const double d = (cv::determinant(U * Vt) < 0.0) ? -1.0 : 1.0;
    cv::Mat D = cv::Mat::eye(3, 3, CV_64F);
    D.at<double>(2, 2) = d;
    const cv::Mat R_mat = U * D * Vt;
    R_world_to_camera = cv::Matx33d(R_mat);

    t_world_to_camera = col2 * lambda;
    return true;
}

// Standard ZYX (yaw-pitch-roll) Euler extraction, R = Rz(yaw)*Ry(pitch)*Rx(roll): pitch = the
// rotated x-axis's elevation out of the horizontal plane; roll = rotation about that axis. This is
// a diagnostic against tape-measure/inclinometer readings, not a control-loop convention -- what
// matters is that it's documented and self-consistent, not that it matches any particular robotics
// library's sign convention.
inline void rotation_to_pitch_roll_deg(const cv::Matx33d& R, double& pitch_deg, double& roll_deg) {
    const double pitch = -std::asin(std::clamp(R(2, 0), -1.0, 1.0));
    const double roll = std::atan2(R(2, 1), R(2, 2));
    pitch_deg = pitch * 180.0 / CV_PI;
    roll_deg = roll * 180.0 / CV_PI;
}

// Composes camera<->ground (from the homography) with the solved radar<->camera transform to get
// radar<->ground, per §9.2. `homography` is the existing bev::HomographyData loaded via
// bev::load_homography_yaml -- its homography_matrix maps rectified-image PIXELS to WORLD meters
// (see calibration_io.hpp), i.e. it is the *inverse* of the world->pixel form the classical
// H=K[r1 r2 t] decomposition expects, so it's inverted here first.
inline GroundPlaneCrossCheck ground_plane_cross_check(
    const bev::HomographyData& homography, const cv::Matx33d& camera_matrix, const RigidTransform& radar_to_camera) {
    GroundPlaneCrossCheck out;

    const cv::Matx33d H_pixel_to_world(homography.homography_matrix);
    const cv::Matx33d H_world_to_pixel = H_pixel_to_world.inv();

    cv::Matx33d R_wc;
    cv::Vec3d t_wc;
    if (!decompose_ground_homography(H_world_to_pixel, camera_matrix, R_wc, t_wc, out.error)) {
        return out;
    }

    // p_ground = R_wc^T * (p_camera - t_wc); substituting p_camera = R_rc*q_radar + t_rc gives the
    // composed radar->ground transform directly.
    const cv::Matx33d R_radar_to_ground = R_wc.t() * radar_to_camera.R;
    const cv::Vec3d t_radar_to_ground = R_wc.t() * (radar_to_camera.t - t_wc);

    // t_radar_to_ground is exactly the radar's own origin (q_radar=0) expressed in ground/world
    // coordinates -- its z-component (world z=0 is the ground plane, per HomographyTab's own
    // "camera_height_m = abs(camera_center.z)" convention) is the derived mounting height.
    out.derived_radar_height_m = std::abs(t_radar_to_ground[2]);
    rotation_to_pitch_roll_deg(R_radar_to_ground, out.radar_pitch_deg, out.radar_roll_deg);
    out.success = true;
    return out;
}

// ---------------------------------------------------------------------------------------------
// §9.3 Pre-solve capture-distribution diagnostic
// ---------------------------------------------------------------------------------------------

struct DistributionDiagnostic {
    bool valid = false;
    double singular_value_ratio = 0.0;  // smallest / largest, of the centered position cloud
    bool degenerate_warning = false;    // true if ratio < svd_ratio_warn_threshold
    double range_spread_m = 0.0;
    double azimuth_spread_deg = 0.0;
    double height_spread_m = 0.0;
    bool height_spread_flagged = false;  // "the one operators forget" -- explicit low-height-spread flag
};

// `planar`: a 2D radar's positions all have z=0 and its planar solve doesn't use height, so the SVD
// degeneracy check runs on x/y only (flags near-collinear captures) and height is never flagged.
inline DistributionDiagnostic compute_capture_distribution_diagnostic(
    const std::vector<cv::Point3d>& radar_positions, double svd_ratio_warn_threshold = 0.15,
    double height_spread_min_m = 0.3, bool planar = false) {
    DistributionDiagnostic diag;
    if (radar_positions.size() < 3) return diag;

    cv::Point3d centroid(0.0, 0.0, 0.0);
    for (const auto& p : radar_positions) centroid += p;
    centroid *= (1.0 / static_cast<double>(radar_positions.size()));

    const int dims = planar ? 2 : 3;
    cv::Mat data(static_cast<int>(radar_positions.size()), dims, CV_64F);
    for (size_t i = 0; i < radar_positions.size(); ++i) {
        data.at<double>(static_cast<int>(i), 0) = radar_positions[i].x - centroid.x;
        data.at<double>(static_cast<int>(i), 1) = radar_positions[i].y - centroid.y;
        if (!planar) data.at<double>(static_cast<int>(i), 2) = radar_positions[i].z - centroid.z;
    }
    cv::Mat U, S, Vt;
    cv::SVD::compute(data, S, U, Vt, cv::SVD::FULL_UV);
    const double largest = S.at<double>(0);
    const double smallest = S.at<double>(dims - 1);
    diag.singular_value_ratio = (largest > 1e-12) ? (smallest / largest) : 0.0;
    diag.degenerate_warning = diag.singular_value_ratio < svd_ratio_warn_threshold;

    double min_range = 1e18, max_range = -1e18;
    double min_az = 1e18, max_az = -1e18;
    double min_z = 1e18, max_z = -1e18;
    for (const auto& p : radar_positions) {
        const double r = cv::norm(p);
        const double az = std::atan2(p.y, p.x) * 180.0 / CV_PI;
        min_range = std::min(min_range, r);
        max_range = std::max(max_range, r);
        min_az = std::min(min_az, az);
        max_az = std::max(max_az, az);
        min_z = std::min(min_z, p.z);
        max_z = std::max(max_z, p.z);
    }
    diag.range_spread_m = max_range - min_range;
    diag.azimuth_spread_deg = max_az - min_az;
    diag.height_spread_m = max_z - min_z;
    diag.height_spread_flagged = !planar && diag.height_spread_m < height_spread_min_m;
    diag.valid = true;
    return diag;
}

}  // namespace radarcam
}  // namespace bev
