#pragma once

// Reduced-DOF (yaw + in-plane translation) radar->camera solve for a 2D radar that reports only
// x/y in its own mounting plane (e.g. DRWIG LRR Pro, drwig_decoder.hpp) -- the "optional DOF mask"
// gate_iterate.hpp's §11 note anticipates.
//
// A 2D radar's data carries no information about pitch, roll or mounting height: every detection
// sits at z=0 whatever the reflector's true height. So those three stay at the measured coarse
// extrinsic (R0, t0) and only what the radar can observe is solved:
//
//   a_i = R0^T * (p_camera_i - t0)            camera-side reflector, in the nominal radar frame
//   a_i.xy ~= Rz(yaw) * q_i.xy + d.xy          2D rigid fit, closed form (2D Kabsch)
//   => R = R0 * Rz(yaw),  t = t0 + R0 * (d.x, d.y, 0)
//
// a_i.z (the reflector's height above/below the radar plane) is deliberately ignored, so the
// reflector can be placed at any height. Residuals are horizontal (in-radar-plane) distances.

#include "bev/radarcam/kabsch.hpp"
#include "bev/radarcam/ransac.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace bev {
namespace radarcam {

// Camera-frame point expressed in the radar frame of `T` (inverse transform).
inline cv::Point3d camera_to_radar_frame(const cv::Point3d& p_camera, const RigidTransform& T) {
    const cv::Vec3d q = T.R.t() * (cv::Vec3d(p_camera.x, p_camera.y, p_camera.z) - T.t);
    return cv::Point3d(q[0], q[1], q[2]);
}

// Fills a planar radar point's missing height from its camera-side partner, so the usual 3D
// residual/reprojection code (R*q+t vs p_camera) measures only the in-plane error the radar can
// actually observe. Use this for every residual/statistic on a planar solve.
inline cv::Point3d complete_planar_point(const cv::Point3d& q_radar, const cv::Point3d& p_camera, const RigidTransform& T) {
    return cv::Point3d(q_radar.x, q_radar.y, camera_to_radar_frame(p_camera, T).z);
}

// In-plane residual of one correspondence under `T`.
inline double planar_residual(const cv::Point3d& p_camera, const cv::Point3d& q_radar, const RigidTransform& T) {
    const cv::Point3d a = camera_to_radar_frame(p_camera, T);
    return std::hypot(a.x - q_radar.x, a.y - q_radar.y);
}

// Least-squares yaw + in-plane offset on top of the fixed base (R0, t0). Needs >=2 correspondences
// with distinct radar x/y positions; returns false (leaving `out` untouched) otherwise.
inline bool solve_planar(const std::vector<cv::Point3d>& p_camera, const std::vector<cv::Point3d>& q_radar,
    const cv::Matx33d& R0, const cv::Vec3d& t0, RigidTransform& out) {
    const size_t n = p_camera.size();
    if (n != q_radar.size() || n < 2) return false;

    const RigidTransform base{R0, t0};
    std::vector<cv::Point2d> a(n), b(n);
    cv::Point2d ca(0.0, 0.0), cb(0.0, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const cv::Point3d a3 = camera_to_radar_frame(p_camera[i], base);
        a[i] = cv::Point2d(a3.x, a3.y);
        b[i] = cv::Point2d(q_radar[i].x, q_radar[i].y);
        ca += a[i];
        cb += b[i];
    }
    ca *= 1.0 / static_cast<double>(n);
    cb *= 1.0 / static_cast<double>(n);

    // Optimal 2D rotation: yaw = atan2(sum cross(b', a'), sum dot(b', a')) over centered points.
    double sum_dot = 0.0, sum_cross = 0.0, spread = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const cv::Point2d A = a[i] - ca;
        const cv::Point2d B = b[i] - cb;
        sum_dot += B.x * A.x + B.y * A.y;
        sum_cross += B.x * A.y - B.y * A.x;
        spread += B.x * B.x + B.y * B.y;
    }
    if (spread < 1e-12) return false;  // all radar points coincide -- yaw is unobservable

    const double yaw = std::atan2(sum_cross, sum_dot);
    const double c = std::cos(yaw), s = std::sin(yaw);
    const cv::Point2d d(ca.x - (c * cb.x - s * cb.y), ca.y - (s * cb.x + c * cb.y));

    const cv::Matx33d Rz(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    out.R = R0 * Rz;
    out.t = t0 + R0 * cv::Vec3d(d.x, d.y, 0.0);
    return true;
}

// RANSAC over solve_planar, same exhaustive-enumeration rationale as solve_ransac (ransac.hpp) --
// but a 2D rigid fit only needs 2 points per sample, and inliers are judged by in-plane residual.
inline RansacResult solve_ransac_planar(const std::vector<cv::Point3d>& p_camera, const std::vector<cv::Point3d>& q_radar,
    const cv::Matx33d& R0, const cv::Vec3d& t0, double inlier_threshold_m) {
    RansacResult result;
    const size_t n = p_camera.size();
    if (n != q_radar.size() || n < 3) return result;  // 2 would fit exactly -- no consensus to measure

    std::vector<int> best_inliers;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            RigidTransform candidate;
            if (!solve_planar({p_camera[i], p_camera[j]}, {q_radar[i], q_radar[j]}, R0, t0, candidate)) continue;

            std::vector<int> inliers;
            inliers.reserve(n);
            for (size_t m = 0; m < n; ++m) {
                if (planar_residual(p_camera[m], q_radar[m], candidate) <= inlier_threshold_m) {
                    inliers.push_back(static_cast<int>(m));
                }
            }
            if (inliers.size() > best_inliers.size()) best_inliers = std::move(inliers);
        }
    }
    if (best_inliers.size() < 3) return result;

    std::vector<cv::Point3d> inlier_p, inlier_q;
    for (int idx : best_inliers) {
        inlier_p.push_back(p_camera[idx]);
        inlier_q.push_back(q_radar[idx]);
    }
    RigidTransform refit;
    if (!solve_planar(inlier_p, inlier_q, R0, t0, refit)) return result;

    result.success = true;
    result.transform = refit;
    result.inlier_indices = std::move(best_inliers);
    return result;
}

}  // namespace radarcam
}  // namespace bev
