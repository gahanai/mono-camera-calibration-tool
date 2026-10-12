#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/clutter_filter.hpp"
#include "bev/radarcam/config.hpp"
#include "bev/radarcam/drwig_decoder.hpp"
#include "bev/radarcam/gate_iterate.hpp"
#include "bev/radarcam/planar_solve.hpp"
#include "bev/radarcam/validate.hpp"

#include <opencv2/calib3d.hpp>

#include <array>
#include <cstdio>
#include <fstream>
#include <string>

using Catch::Matchers::WithinAbs;
using namespace bev::radarcam;

namespace {

using Frame = std::array<uint8_t, 8>;

Frame hex8(const std::string& hex) {
    Frame f{};
    for (size_t i = 0; i < 8; ++i) f[i] = static_cast<uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    return f;
}

cv::Matx33d rodrigues(double rx, double ry, double rz) {
    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx, ry, rz);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}

cv::Matx33d rot_z(double yaw) {
    const double c = std::cos(yaw), s = std::sin(yaw);
    return cv::Matx33d(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
}

cv::Point3d apply(const RigidTransform& T, const cv::Point3d& q) {
    const cv::Vec3d v = T.R * cv::Vec3d(q.x, q.y, q.z) + T.t;
    return cv::Point3d(v[0], v[1], v[2]);
}

void check_transform_near(const RigidTransform& got, const RigidTransform& want, double tol) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) CHECK_THAT(got.R(r, c), WithinAbs(want.R(r, c), tol));
        CHECK_THAT(got.t[r], WithinAbs(want.t[r], tol));
    }
}

// A rear-radar-like mount (radar looking back and to the side of a rear camera) plus a small
// yaw/offset error the solve has to find -- mirrors how the coarse config relates to the truth.
struct PlanarScenario {
    cv::Matx33d R0 = rodrigues(-1.2, 0.9, -1.4);
    cv::Vec3d t0{0.75, 0.10, 0.0};
    RigidTransform truth;
    std::vector<cv::Point3d> reflector_radar3d;  // true reflector positions incl. height off the radar plane

    PlanarScenario() {
        truth.R = R0 * rot_z(3.0 * CV_PI / 180.0);
        truth.t = t0 + R0 * cv::Vec3d(0.12, -0.08, 0.0);
        for (int i = 0; i < 12; ++i) {
            reflector_radar3d.emplace_back(4.0 + i * 1.0, -4.0 + (i % 5) * 2.0, -0.3 + (i % 4) * 0.3);
        }
    }
    cv::Point3d camera_point(size_t i) const { return apply(truth, reflector_radar3d[i]); }
    cv::Point3d radar_point(size_t i) const {  // what a 2D radar reports: height lost
        return cv::Point3d(reflector_radar3d[i].x, reflector_radar3d[i].y, 0.0);
    }
};

}  // namespace

// Real frames captured from the vehicle's can1 bus (rear-left radar, radarId 4 -> ID offset 0x530).
TEST_CASE("DRWIG decoder decodes real rear-left radar frames", "[drwig][decoder]") {
    const Frame p2_slot1 = hex8("800F590681800779");  // 0x551
    const Frame p1_slot1 = hex8("0183815F8013D7C5");  // 0x581
    const Frame status = hex8("381C5CC82002A78B");    // 0x5B0

    for (const Frame& f : {p2_slot1, p1_slot1, status}) {
        CHECK(drwig::crc8_sae_j1850_h2f(f.data(), 7) == f[7]);
    }

    DrwigLrrProDecoder decoder(4);
    CHECK_FALSE(decoder.handle_frame(0x551, p2_slot1.data(), 8, 0));
    CHECK_FALSE(decoder.handle_frame(0x581, p1_slot1.data(), 8, 10));
    REQUIRE(decoder.handle_frame(0x5B0, status.data(), 8, 20));

    REQUIRE(decoder.detections().size() == 1);
    const auto& d = decoder.detections()[0];
    CHECK_THAT(d.position.x, WithinAbs(6.5, 1e-9));
    CHECK_THAT(d.position.y, WithinAbs(-2.609375, 1e-9));  // raw DBC sign: negative = radar's left
    CHECK_THAT(d.position.z, WithinAbs(0.0, 1e-12));
    CHECK_THAT(d.range_m, WithinAbs(std::hypot(6.5, -2.609375), 1e-9));
    CHECK_THAT(d.elevation_rad, WithinAbs(0.0, 1e-12));
    CHECK_THAT(d.vel_mps, WithinAbs(0.0, 1e-9));
    CHECK(d.is_stationary);  // motion pattern 2 = stopped
    CHECK(decoder.frame_id() == 1);
    CHECK(decoder.bad_checksum() == 0);

    SECTION("stale slots are dropped on the next cycle") {
        REQUIRE(decoder.handle_frame(0x5B0, status.data(), 8, 20 + DrwigLrrProDecoder::kStaleTimeoutMs + 1));
        CHECK(decoder.detections().empty());
    }
}

TEST_CASE("DRWIG decoder ignores other radars and rejects bad checksums", "[drwig][decoder]") {
    const Frame p2_slot1 = hex8("800F590681800779");
    const Frame status = hex8("381C5CC82002A78B");

    SECTION("a decoder for radarId 5 ignores radarId 4's frames") {
        DrwigLrrProDecoder decoder(5);
        CHECK_FALSE(decoder.handle_frame(0x551, p2_slot1.data(), 8, 0));
        CHECK_FALSE(decoder.handle_frame(0x5B0, status.data(), 8, 10));
        CHECK(decoder.frame_id() == 0);
        CHECK(decoder.bad_checksum() == 0);
    }

    SECTION("a corrupted payload is counted and not decoded") {
        DrwigLrrProDecoder decoder(4);
        Frame corrupted = p2_slot1;
        corrupted[3] ^= 0x10;
        CHECK_FALSE(decoder.handle_frame(0x551, corrupted.data(), 8, 0));
        CHECK(decoder.bad_checksum() == 1);
    }

    SECTION("slots whose Part1 says invalid are not published") {
        DrwigLrrProDecoder decoder(4);
        Frame p1_invalid = hex8("0183815F8013D7C5");
        p1_invalid[2] &= 0x7F;  // Obj_ValidFlag = start bit 23 -> byte 2, bit 7
        p1_invalid[7] = drwig::crc8_sae_j1850_h2f(p1_invalid.data(), 7);
        decoder.handle_frame(0x551, p2_slot1.data(), 8, 0);
        decoder.handle_frame(0x581, p1_invalid.data(), 8, 10);
        REQUIRE(decoder.handle_frame(0x5B0, status.data(), 8, 20));
        CHECK(decoder.detections().empty());
    }
}

TEST_CASE("planar solve recovers yaw + in-plane offset despite reflector height", "[drwig][planar]") {
    const PlanarScenario s;
    std::vector<cv::Point3d> p, q;
    for (size_t i = 0; i < s.reflector_radar3d.size(); ++i) {
        p.push_back(s.camera_point(i));
        q.push_back(s.radar_point(i));
    }

    RigidTransform solved;
    REQUIRE(solve_planar(p, q, s.R0, s.t0, solved));
    check_transform_near(solved, s.truth, 1e-9);
    for (size_t i = 0; i < p.size(); ++i) {
        CHECK_THAT(planar_residual(p[i], q[i], solved), WithinAbs(0.0, 1e-9));
        // Completing the missing height makes the ordinary 3D residual vanish too.
        CHECK_THAT(cv::norm(apply(solved, complete_planar_point(q[i], p[i], solved)) - p[i]), WithinAbs(0.0, 1e-9));
    }

    SECTION("RANSAC rejects outliers") {
        for (int i : {1, 5, 9}) q[i].x += 1.5;
        const auto result = solve_ransac_planar(p, q, s.R0, s.t0, 0.25);
        REQUIRE(result.success);
        CHECK(result.inlier_indices.size() == p.size() - 3);
        check_transform_near(result.transform, s.truth, 1e-9);
    }

    SECTION("coincident radar points leave yaw unobservable") {
        const std::vector<cv::Point3d> same(3, q[0]);
        RigidTransform untouched;
        CHECK_FALSE(solve_planar({p[0], p[1], p[2]}, same, s.R0, s.t0, untouched));
    }
}

TEST_CASE("clutter filter planar mode gates in the radar plane and works without RCS", "[drwig][clutter]") {
    // Reflector 0.5m above the radar plane: a 2D radar still reports it at z=0.
    const cv::Point3d q_hat(8.0, 1.0, 0.5);
    RadarDwell dwell;
    for (int f = 0; f < 10; ++f) {
        RadarDetection near, far;
        near.position = cv::Point3d(8.1, 1.0, 0.0);
        far.position = cv::Point3d(8.3, 1.0, 0.0);
        dwell.frames.push_back({far, near});  // rcs_dbsm = 0 for both
    }

    ClutterFilterParams params;
    params.gate_radius_m = 0.4;
    params.sigma_trihedral_dbsm = 20.0;  // expected ~-16 dBsm at 8m, so rcs=0 would fail a real RCS gate

    SECTION("3D gating rejects it on height alone") {
        const auto r = run_clutter_pipeline(dwell, {}, false, q_hat, params);
        CHECK_FALSE(r.accepted);
    }

    SECTION("planar + no-RCS accepts it and picks the detection nearest the prediction") {
        params.planar = true;
        params.has_rcs = false;
        const auto r = run_clutter_pipeline(dwell, {}, false, q_hat, params);
        REQUIRE(r.accepted);
        CHECK_THAT(r.q_radar.x, WithinAbs(8.1, 1e-12));
        CHECK_THAT(r.q_radar.y, WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("gate iterate solves a planar radar end-to-end from dwells", "[drwig][gate_iterate]") {
    const PlanarScenario s;
    std::vector<CaptureInput> captures;
    for (size_t i = 0; i < s.reflector_radar3d.size(); ++i) {
        CaptureInput ci;
        ci.p_camera = s.camera_point(i);
        for (int f = 0; f < 25; ++f) {
            RadarDetection reflector, clutter;
            reflector.position = s.radar_point(i);
            clutter.position = s.radar_point(i) + cv::Point3d(0.0, 3.0, 0.0);
            ci.dwell.frames.push_back({clutter, reflector});
        }
        captures.push_back(ci);
    }

    ClutterFilterParams params;
    params.planar = true;
    params.has_rcs = false;
    const auto result =
        run_gate_iterate(captures, {2.0, 0.5, 0.4}, s.R0, s.t0, params, 0.25, 0.5, /*planar=*/true);

    REQUIRE(result.success);
    CHECK(result.final_inlier_indices.size() == captures.size());
    check_transform_near(result.transform, s.truth, 1e-9);
}

TEST_CASE("distribution diagnostic planar mode checks x/y spread only", "[drwig][validate]") {
    std::vector<cv::Point3d> spread_xy, collinear;
    for (int i = 0; i < 6; ++i) {
        spread_xy.emplace_back(5.0 + i, (i % 3) * 2.0 - 2.0, 0.0);
        collinear.emplace_back(5.0 + i, 0.0, 0.0);
    }

    const auto good = compute_capture_distribution_diagnostic(spread_xy, 0.15, 0.3, /*planar=*/true);
    CHECK_FALSE(good.degenerate_warning);
    CHECK_FALSE(good.height_spread_flagged);

    const auto bad = compute_capture_distribution_diagnostic(collinear, 0.15, 0.3, /*planar=*/true);
    CHECK(bad.degenerate_warning);

    const auto as_3d = compute_capture_distribution_diagnostic(spread_xy, 0.15, 0.3);
    CHECK(as_3d.degenerate_warning);  // 3D mode can't tell a flat 2D radar cloud from a degenerate one
    CHECK(as_3d.height_spread_flagged);
}

TEST_CASE("radarcam config reads the radar type fields", "[drwig][config]") {
    const std::string path = std::string(FIXTURES_DIR) + "/radar_type_config_scratch.yaml";
    auto load = [&](const std::string& yaml, RadarCamConfig& cfg, std::string& error) {
        {
            std::ofstream fs(path, std::ios::trunc);
            fs << yaml;
        }
        const bool ok = load_radarcam_config_yaml(path, cfg, error);
        std::remove(path.c_str());
        return ok;
    };

    RadarCamConfig defaults;
    CHECK(defaults.radar_type == "columbus_fc4d");
    CHECK_FALSE(defaults.is_drwig());

    RadarCamConfig cfg;
    std::string error;
    REQUIRE(load("radar_type: drwig_lrr_pro\ndrwig_radar_id: 4\ncan_interface: can1\n", cfg, error));
    CHECK(cfg.is_drwig());
    CHECK(cfg.drwig_radar_id == 4);
    CHECK(cfg.can_interface == "can1");

    RadarCamConfig bad_type;
    CHECK_FALSE(load("radar_type: drwig\n", bad_type, error));
    CHECK_FALSE(error.empty());

    RadarCamConfig bad_id;
    CHECK_FALSE(load("radar_type: drwig_lrr_pro\ndrwig_radar_id: 7\n", bad_id, error));
}
