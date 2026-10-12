#pragma once

// Radar-camera calibration config -- every threshold the calibration pipeline uses, loaded
// from YAML via yaml-cpp so nothing in radarcam/* hard-codes a magic number (radcam_calibplan.md
// §4/§12 item 4). Missing keys keep the struct's default below rather than failing to load, so a
// partial config YAML during early tuning doesn't hard-error.

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "bev/config_types.hpp"

namespace bev {
namespace radarcam {

struct RadarCamConfig {
    // Which radar this config calibrates. "columbus_fc4d" (default): CAN-FD 3D raw detections with RCS,
    // full 6-DOF solve. "drwig_lrr_pro": classic CAN 2D tracked objects, no RCS/elevation -- the RCS
    // gate is skipped and the solve is planar (yaw + in-plane offset only; pitch/roll/height stay at
    // the coarse values below, see planar_solve.hpp).
    std::string radar_type = "columbus_fc4d";
    int drwig_radar_id = 0;     // radarId 0..5 selecting one radar on a shared bus (4 = rear-left, 5 = rear-right)
    std::string can_interface;  // optional; when set, overrides the tab's CAN Interface field on load

    bool is_drwig() const { return radar_type == "drwig_lrr_pro"; }

    // Target rig geometry (§2) -- reuses the same checkerboard params as intrinsics calibration.
    ChessboardConfig board;
    cv::Vec3d x_b_measured_m{0.0, 0.0, 0.0};  // reflector phase-center offset in board frame (calipers)

    // Coarse extrinsic bootstrap (§2) -- tape measure + inclinometer, seeds the first (wide) spatial gate.
    cv::Vec3d coarse_euler_deg{0.0, 0.0, 0.0};  // roll, pitch, yaw -- radar -> camera
    cv::Vec3d coarse_translation_m{0.0, 0.0, 0.0};

    // Camera-side quality gates (§5)
    double pnp_reproj_rms_max_px = 1.0;
    double board_min_tilt_deg = 15.0;

    // Radar clutter filter thresholds (§6)
    double doppler_threshold_mps = 0.5;
    double sigma_trihedral_dbsm = 20.0;  // known trihedral RCS at r_ref_m -- set from reflector geometry
    double r_ref_m = 1.0;
    double rcs_margin_db = 10.0;
    double background_match_radius_m = 0.3;   // stage 0: proximity match vs. background scan
    double radar_height_above_ground_m = 0.5;  // stage 4: tape-measured, for the ground-plausibility gate
    double persistence_frac = 0.8;
    double spread_max_m = 0.10;

    // Solver (§7)
    std::vector<double> gate_radii_m{2.0, 0.5, 0.4};  // per-iteration spatial gate radius, wide -> tight
    double ransac_inlier_threshold_m = 0.25;
    int max_gate_iterations = 3;
    double inlier_drop_warn_frac = 0.5;  // warn if inliers drop by more than this fraction between iterations

    // Nonlinear refinement (§8)
    std::string refine_mode = "B";      // "A" or "B" -- which mode's result is the primary answer
    bool run_mode_b_diagnostic = true;  // spec: always run Mode B as a diagnostic, regardless of refine_mode
    double huber_delta_m = 0.1;

    // Validation (§9)
    double holdout_fraction = 0.2;
    double svd_ratio_warn_threshold = 0.15;
};

inline bool load_radarcam_config_yaml(const std::string& path, RadarCamConfig& out, std::string& error) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        error = std::string("Failed to load/parse YAML: ") + e.what();
        return false;
    }

    auto read_double = [&](const char* key, double& v) {
        if (root[key]) v = root[key].as<double>();
    };
    auto read_int = [&](const char* key, int& v) {
        if (root[key]) v = root[key].as<int>();
    };
    auto read_string = [&](const char* key, std::string& v) {
        if (root[key]) v = root[key].as<std::string>();
    };
    auto read_bool = [&](const char* key, bool& v) {
        if (root[key]) v = root[key].as<bool>();
    };
    auto read_vec3 = [&](const char* key, cv::Vec3d& v) {
        if (root[key] && root[key].IsSequence() && root[key].size() == 3) {
            v = cv::Vec3d(root[key][0].as<double>(), root[key][1].as<double>(), root[key][2].as<double>());
        }
    };

    read_string("radar_type", out.radar_type);
    if (out.radar_type != "columbus_fc4d" && out.radar_type != "drwig_lrr_pro") {
        error = "radar_type must be \"columbus_fc4d\" or \"drwig_lrr_pro\", got \"" + out.radar_type + "\".";
        return false;
    }
    read_int("drwig_radar_id", out.drwig_radar_id);
    if (out.drwig_radar_id < 0 || out.drwig_radar_id > 5) {
        error = "drwig_radar_id must be 0..5, got " + std::to_string(out.drwig_radar_id) + ".";
        return false;
    }
    read_string("can_interface", out.can_interface);

    if (root["board"]) {
        const auto& b = root["board"];
        if (b["inner_corners_x"]) out.board.inner_corners_x = b["inner_corners_x"].as<int>();
        if (b["inner_corners_y"]) out.board.inner_corners_y = b["inner_corners_y"].as<int>();
        if (b["square_size_m"]) out.board.square_size_m = b["square_size_m"].as<double>();
    }

    read_vec3("x_b_measured_m", out.x_b_measured_m);
    read_vec3("coarse_euler_deg", out.coarse_euler_deg);
    read_vec3("coarse_translation_m", out.coarse_translation_m);

    read_double("pnp_reproj_rms_max_px", out.pnp_reproj_rms_max_px);
    read_double("board_min_tilt_deg", out.board_min_tilt_deg);

    read_double("doppler_threshold_mps", out.doppler_threshold_mps);
    read_double("sigma_trihedral_dbsm", out.sigma_trihedral_dbsm);
    read_double("r_ref_m", out.r_ref_m);
    read_double("rcs_margin_db", out.rcs_margin_db);
    read_double("background_match_radius_m", out.background_match_radius_m);
    read_double("radar_height_above_ground_m", out.radar_height_above_ground_m);
    read_double("persistence_frac", out.persistence_frac);
    read_double("spread_max_m", out.spread_max_m);

    if (root["gate_radii_m"] && root["gate_radii_m"].IsSequence()) {
        out.gate_radii_m.clear();
        for (const auto& n : root["gate_radii_m"]) {
            out.gate_radii_m.push_back(n.as<double>());
        }
    }
    read_double("ransac_inlier_threshold_m", out.ransac_inlier_threshold_m);
    read_int("max_gate_iterations", out.max_gate_iterations);
    read_double("inlier_drop_warn_frac", out.inlier_drop_warn_frac);

    read_string("refine_mode", out.refine_mode);
    read_bool("run_mode_b_diagnostic", out.run_mode_b_diagnostic);
    read_double("huber_delta_m", out.huber_delta_m);

    read_double("holdout_fraction", out.holdout_fraction);
    read_double("svd_ratio_warn_threshold", out.svd_ratio_warn_threshold);

    return true;
}

}  // namespace radarcam
}  // namespace bev
