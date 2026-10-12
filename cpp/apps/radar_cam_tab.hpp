#pragma once

// Radar-camera extrinsic calibration tab. IMPORTANT: this file is a textual #include INTO
// calibration_tool_qt.cpp, placed AFTER that file's shared helpers (mat_to_qimage,
// enumerate_video_indices_linux, make_toggle_button, set_toggle_button_visual, set_state_label,
// ClickableLabel, default_camera_asset_dir, load_intrinsics_yaml) and BEFORE main() -- it is not
// meant to be included standalone. The algorithmic core it orchestrates lives in the Qt-free
// bev::radarcam headers under include/bev/radarcam/, so that core stays unit-testable without Qt;
// this file only wires that core into the same tab/UI conventions IntrinsicsTab and HomographyTab
// already establish (sidebar config/paths + view finder, single status_label_, centralized
// apply_ui_state(), make_toggle_button() for Preview/Radar-Live toggles).

#include "bev/radarcam/board_pnp.hpp"
#include "bev/radarcam/capture_io.hpp"
#include "bev/radarcam/clutter_filter.hpp"
#include "bev/radarcam/config.hpp"
#include "bev/radarcam/diagnostics_render.hpp"
#include "bev/radarcam/extrinsics_io.hpp"
#include "bev/radarcam/gate_iterate.hpp"
#include "bev/radarcam/refine_ceres.hpp"
#include "bev/radarcam/socketcan_reader.hpp"
#include "bev/radarcam/validate.hpp"

#include <opencv2/calib3d.hpp>

#include <filesystem>

namespace radar_cam_tab_detail {

inline cv::Matx33d axis_angle_to_matx(double rx, double ry, double rz) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << rx, ry, rz);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}

// Standard ZYX (yaw-pitch-roll, extrinsic) composition: R = Rz(yaw)*Ry(pitch)*Rx(roll). Matches
// the convention bev::radarcam::rotation_to_pitch_roll_deg's extraction assumes.
inline cv::Matx33d euler_deg_to_rotation(const cv::Vec3d& euler_deg) {
    const double d2r = 3.14159265358979323846 / 180.0;
    const cv::Matx33d Rx = axis_angle_to_matx(euler_deg[0] * d2r, 0.0, 0.0);
    const cv::Matx33d Ry = axis_angle_to_matx(0.0, euler_deg[1] * d2r, 0.0);
    const cv::Matx33d Rz = axis_angle_to_matx(0.0, 0.0, euler_deg[2] * d2r);
    return Rz * Ry * Rx;
}

inline std::vector<std::string> list_capture_dirs(const std::string& base_dir) {
    std::vector<std::string> dirs;
    if (!std::filesystem::exists(base_dir)) return dirs;
    for (const auto& entry : std::filesystem::directory_iterator(base_dir)) {
        if (entry.is_directory() && entry.path().filename().string().rfind("capture_", 0) == 0) {
            dirs.push_back(entry.path().string());
        }
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

inline int find_next_capture_index(const std::string& base_dir) {
    int max_idx = 0;
    for (const auto& d : list_capture_dirs(base_dir)) {
        const std::string name = std::filesystem::path(d).filename().string();
        try {
            max_idx = std::max(max_idx, std::stoi(name.substr(8)));
        } catch (const std::exception&) {
        }
    }
    return max_idx + 1;
}

inline bev::radarcam::ClutterFunnel aggregate_funnel(const std::vector<bev::radarcam::ClutterFilterResult>& per_capture) {
    bev::radarcam::ClutterFunnel agg = {bev::radarcam::StageFunnelCount{"background_subtraction", 0, 0},
        bev::radarcam::StageFunnelCount{"spatial_gate", 0, 0}, bev::radarcam::StageFunnelCount{"doppler_gate", 0, 0},
        bev::radarcam::StageFunnelCount{"rcs_gate", 0, 0}, bev::radarcam::StageFunnelCount{"ground_plausibility", 0, 0},
        bev::radarcam::StageFunnelCount{"best_in_gate", 0, 0}, bev::radarcam::StageFunnelCount{"persistence_aggregation", 0, 0}};
    for (const auto& r : per_capture) {
        for (size_t i = 0; i < agg.size(); ++i) {
            agg[i].n_entering += r.funnel[i].n_entering;
            agg[i].n_surviving += r.funnel[i].n_surviving;
        }
    }
    return agg;
}

}  // namespace radar_cam_tab_detail

class RadarCamTab final : public QWidget {
public:
    explicit RadarCamTab(QWidget* parent = nullptr) : QWidget(parent) {
        auto* root_layout = new QHBoxLayout(this);
        root_layout->setSpacing(14);

        auto* left_panel = new QGroupBox("Radar-Camera Calibration");
        auto* left_form = new QFormLayout(left_panel);
        left_form->setLabelAlignment(Qt::AlignLeft);
        left_form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

        camera_combo_ = new QComboBox();
        refresh_btn_ = new QPushButton("Refresh");
        auto* cam_row = new QWidget();
        auto* cam_layout = new QHBoxLayout(cam_row);
        cam_layout->setContentsMargins(0, 0, 0, 0);
        cam_layout->addWidget(camera_combo_, 1);
        cam_layout->addWidget(refresh_btn_);

        intrinsics_path_ = new QLineEdit(default_camera_asset_dir(0) + "/intrinsics.yaml");
        auto* browse_intrinsics_btn = new QPushButton("Browse");
        auto* reload_intrinsics_btn = new QPushButton("Reload");
        auto* intrinsics_row = new QWidget();
        auto* intrinsics_layout = new QHBoxLayout(intrinsics_row);
        intrinsics_layout->setContentsMargins(0, 0, 0, 0);
        intrinsics_layout->addWidget(intrinsics_path_, 1);
        intrinsics_layout->addWidget(browse_intrinsics_btn);
        intrinsics_layout->addWidget(reload_intrinsics_btn);

        homography_path_ = new QLineEdit(default_camera_asset_dir(0) + "/homography.yaml");
        homography_path_->setToolTip("Optional -- only needed for the §9.2 ground-plane cross-check in Validate.");
        auto* browse_homography_btn = new QPushButton("Browse");
        auto* reload_homography_btn = new QPushButton("Reload");
        auto* homography_row = new QWidget();
        auto* homography_layout = new QHBoxLayout(homography_row);
        homography_layout->setContentsMargins(0, 0, 0, 0);
        homography_layout->addWidget(homography_path_, 1);
        homography_layout->addWidget(browse_homography_btn);
        homography_layout->addWidget(reload_homography_btn);

        config_path_ = new QLineEdit(default_camera_asset_dir(0) + "/radar_cam_config.yaml");
        auto* browse_config_btn = new QPushButton("Browse");
        auto* reload_config_btn = new QPushButton("Reload");
        auto* config_row = new QWidget();
        auto* config_layout = new QHBoxLayout(config_row);
        config_layout->setContentsMargins(0, 0, 0, 0);
        config_layout->addWidget(config_path_, 1);
        config_layout->addWidget(browse_config_btn);
        config_layout->addWidget(reload_config_btn);

        can_interface_ = new QLineEdit("can1");
        captures_dir_ = new QLineEdit(default_camera_asset_dir(0) + "/radar_cam_captures");
        output_extrinsics_path_ = new QLineEdit(default_camera_asset_dir(0) + "/radar_extrinsics.yaml");

        left_form->addRow("Camera", cam_row);
        left_form->addRow("Intrinsics YAML", intrinsics_row);
        left_form->addRow("Homography YAML", homography_row);
        left_form->addRow("Radar-Cam Config", config_row);
        left_form->addRow("CAN Interface", can_interface_);
        left_form->addRow("Captures Dir", captures_dir_);
        left_form->addRow("Output Extrinsics", output_extrinsics_path_);

        camera_preview_toggle_ = make_toggle_button("Camera Preview");
        camera_preview_toggle_->setMinimumHeight(34);
        radar_live_toggle_ = make_toggle_button("Radar Live");
        radar_live_toggle_->setMinimumHeight(34);
        live_overlay_toggle_ = make_toggle_button("Live Overlay");
        live_overlay_toggle_->setMinimumHeight(34);
        capture_btn_ = new QPushButton("Capture");
        inspect_btn_ = new QPushButton("Run Inspect (§9.3)");
        solve_btn_ = new QPushButton("Run Solve");
        solve_btn_->setObjectName("primaryButton");
        validate_btn_ = new QPushButton("Run Validate");

        auto* controls = new QGroupBox("Actions");
        auto* controls_layout = new QVBoxLayout(controls);
        controls_layout->setSpacing(10);
        auto* toggle_row = new QWidget();
        auto* toggle_row_layout = new QHBoxLayout(toggle_row);
        toggle_row_layout->setContentsMargins(0, 0, 0, 0);
        toggle_row_layout->addWidget(camera_preview_toggle_);
        toggle_row_layout->addWidget(radar_live_toggle_);
        toggle_row_layout->addWidget(live_overlay_toggle_);
        auto* action_row = new QWidget();
        auto* action_row_layout = new QHBoxLayout(action_row);
        action_row_layout->setContentsMargins(0, 0, 0, 0);
        action_row_layout->addWidget(capture_btn_);
        auto* action_row2 = new QWidget();
        auto* action_row2_layout = new QHBoxLayout(action_row2);
        action_row2_layout->setContentsMargins(0, 0, 0, 0);
        action_row2_layout->addWidget(inspect_btn_);
        action_row2_layout->addWidget(solve_btn_);
        action_row2_layout->addWidget(validate_btn_);
        controls_layout->addWidget(toggle_row);
        controls_layout->addWidget(action_row);
        controls_layout->addWidget(action_row2);

        intrinsics_status_label_ = new QLabel("Intrinsics: not loaded.");
        intrinsics_status_label_->setWordWrap(true);
        config_status_label_ = new QLabel("Config: not loaded (defaults in use).");
        config_status_label_->setWordWrap(true);
        radar_status_label_ = new QLabel("Radar: not connected.");
        radar_status_label_->setWordWrap(true);
        status_label_ = new QLabel(
            "Load intrinsics + a radar-cam config, start Camera Preview and Radar Live, then Capture the target at "
            "varied range/azimuth/height (§10).");
        status_label_->setWordWrap(true);

        auto* left_stack = new QVBoxLayout();
        left_stack->addWidget(left_panel);
        left_stack->addWidget(intrinsics_status_label_);
        left_stack->addWidget(config_status_label_);
        left_stack->addWidget(controls);
        left_stack->addWidget(radar_status_label_);
        left_stack->addWidget(status_label_);
        left_stack->addStretch(1);

        auto* left_container = new QWidget();
        left_container->setLayout(left_stack);
        auto* left_scroll = new QScrollArea();
        left_scroll->setObjectName("sidebarScroll");
        left_scroll->setWidget(left_container);
        left_scroll->setWidgetResizable(true);
        left_scroll->setFrameShape(QFrame::NoFrame);
        left_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        left_scroll->setMinimumWidth(440);

        auto* right_panel = new QGroupBox("View Finder");
        auto* right_layout = new QVBoxLayout(right_panel);
        preview_label_ = new QLabel();
        preview_label_->setObjectName("previewArea");
        preview_label_->setAlignment(Qt::AlignCenter);
        preview_label_->setMinimumSize(640, 360);
        preview_label_->setText("Camera preview stopped.");
        diagnostics_label_ = new QLabel();
        diagnostics_label_->setObjectName("previewArea");
        diagnostics_label_->setAlignment(Qt::AlignCenter);
        diagnostics_label_->setMinimumSize(640, 220);
        diagnostics_label_->setText("Diagnostics will appear here after Inspect/Solve/Validate.");
        right_layout->addWidget(preview_label_, 3);
        right_layout->addWidget(diagnostics_label_, 2);

        root_layout->addWidget(left_scroll, 1);
        root_layout->addWidget(right_panel, 3);

        timer_ = new QTimer(this);
        timer_->setInterval(33);
        connect(timer_, &QTimer::timeout, this, [this]() { on_frame_tick(); });

        connect(refresh_btn_, &QPushButton::clicked, this, [this]() { refresh_cameras(); });
        connect(camera_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int) { update_default_paths(); });
        connect(browse_intrinsics_btn, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getOpenFileName(
                this, "Load Intrinsics YAML", intrinsics_path_->text(), "YAML files (*.yaml *.yml)");
            if (!path.isEmpty()) {
                intrinsics_path_->setText(path);
                load_intrinsics();
            }
        });
        connect(reload_intrinsics_btn, &QPushButton::clicked, this, [this]() { load_intrinsics(); });
        connect(browse_homography_btn, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getOpenFileName(
                this, "Load Homography YAML", homography_path_->text(), "YAML files (*.yaml *.yml)");
            if (!path.isEmpty()) {
                homography_path_->setText(path);
                load_homography();
            }
        });
        connect(reload_homography_btn, &QPushButton::clicked, this, [this]() { load_homography(); });
        connect(browse_config_btn, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getOpenFileName(
                this, "Load Radar-Cam Config YAML", config_path_->text(), "YAML files (*.yaml *.yml)");
            if (!path.isEmpty()) {
                config_path_->setText(path);
                load_config();
            }
        });
        connect(reload_config_btn, &QPushButton::clicked, this, [this]() { load_config(); });
        connect(camera_preview_toggle_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_camera_preview();
            } else {
                stop_camera_preview();
            }
        });
        connect(radar_live_toggle_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_radar_live();
            } else {
                stop_radar_live();
            }
        });
        connect(live_overlay_toggle_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_live_overlay();
            } else {
                stop_live_overlay();
            }
        });
        connect(capture_btn_, &QPushButton::clicked, this, [this]() { begin_capture(); });
        connect(inspect_btn_, &QPushButton::clicked, this, [this]() { run_inspect(); });
        connect(solve_btn_, &QPushButton::clicked, this, [this]() { run_solve(); });
        connect(validate_btn_, &QPushButton::clicked, this, [this]() { run_validate(); });

        refresh_cameras();
        load_config();
        apply_ui_state();
    }

    ~RadarCamTab() override {
        stop_camera_preview();
        stop_radar_live();
    }

private:
    // ---------------------------------------------------------------------------------------
    // Camera / config loading (mirrors HomographyTab's exact conventions)
    // ---------------------------------------------------------------------------------------

    void refresh_cameras() {
        camera_combo_->clear();
        for (const int idx : enumerate_video_indices_linux()) {
            cv::VideoCapture probe(idx, cv::CAP_V4L2);
            if (probe.isOpened()) {
                camera_combo_->addItem(QString("Camera %1").arg(idx), idx);
                probe.release();
            }
        }
        if (camera_combo_->count() == 0) {
            status_label_->setText("No cameras found.");
        }
        update_default_paths();
    }

    void update_default_paths() {
        const int camera_id = camera_combo_->count() > 0 ? camera_combo_->currentData().toInt() : 0;
        const QString dir = default_camera_asset_dir(camera_id);
        intrinsics_path_->setText(dir + "/intrinsics.yaml");
        homography_path_->setText(dir + "/homography.yaml");
        config_path_->setText(dir + "/radar_cam_config.yaml");
        captures_dir_->setText(dir + "/radar_cam_captures");
        output_extrinsics_path_->setText(dir + "/radar_extrinsics.yaml");
        load_intrinsics();
        load_homography();
        load_config();
    }

    void load_intrinsics() {
        bev::IntrinsicsData data;
        QString error;
        if (!load_intrinsics_yaml(intrinsics_path_->text().toStdString(), data, error)) {
            intrinsics_loaded_ = false;
            intrinsics_status_label_->setText(QString("Intrinsics: failed to load (%1)").arg(error));
            apply_ui_state();
            return;
        }
        intrinsics_image_size_ = cv::Size(data.image_width, data.image_height);
        intrinsics_camera_matrix_ = data.camera_matrix;
        intrinsics_dist_coeffs_ = data.dist_coeffs;
        intrinsics_projection_matrix_ = data.projection_matrix;
        intrinsics_loaded_ = true;
        intrinsics_status_label_->setText(
            QString("Intrinsics loaded: %1x%2").arg(data.image_width).arg(data.image_height));
        apply_ui_state();
    }

    void load_homography() {
        bev::HomographyData data;
        std::string error;
        if (!bev::load_homography_yaml(homography_path_->text().toStdString(), data, error)) {
            homography_loaded_ = false;
            return;
        }
        homography_data_ = data;
        homography_loaded_ = true;
    }

    void load_config() {
        // Load into a fresh struct: a key missing from this file must fall back to its default, not
        // keep whatever the previously loaded config set (e.g. radar_type from a rear DRWIG config
        // leaking into a front config that doesn't mention it).
        bev::radarcam::RadarCamConfig loaded;
        std::string error;
        if (!bev::radarcam::load_radarcam_config_yaml(config_path_->text().toStdString(), loaded, error)) {
            config_ = bev::radarcam::RadarCamConfig{};
            config_status_label_->setText(QString("Config: not loaded (%1) -- using struct defaults.")
                .arg(QString::fromStdString(error)));
            apply_ui_state();
            return;
        }
        config_ = loaded;
        if (!config_.can_interface.empty()) {
            can_interface_->setText(QString::fromStdString(config_.can_interface));
        }
        const QString radar_desc = config_.is_drwig()
            ? QString("DRWIG LRR Pro radarId %1, 2D planar solve").arg(config_.drwig_radar_id)
            : QString("Columbus FC 4D, 6-DOF solve");
        config_status_label_->setText(QString("Config loaded: %1; board %2x%3 @ %4m, %5 gate iteration(s).")
            .arg(radar_desc)
            .arg(config_.board.inner_corners_x)
            .arg(config_.board.inner_corners_y)
            .arg(config_.board.square_size_m, 0, 'f', 3)
            .arg(static_cast<int>(config_.gate_radii_m.size())));
        apply_ui_state();
    }

    // Clutter-filter params shared by Inspect/Solve/Validate; callers set gate_radius_m themselves.
    bev::radarcam::ClutterFilterParams clutter_params() const {
        bev::radarcam::ClutterFilterParams params;
        params.doppler_threshold_mps = config_.doppler_threshold_mps;
        params.sigma_trihedral_dbsm = config_.sigma_trihedral_dbsm;
        params.r_ref_m = config_.r_ref_m;
        params.rcs_margin_db = config_.rcs_margin_db;
        params.background_match_radius_m = config_.background_match_radius_m;
        params.radar_height_above_ground_m = config_.radar_height_above_ground_m;
        params.persistence_frac = config_.persistence_frac;
        params.spread_max_m = config_.spread_max_m;
        params.planar = config_.is_drwig();
        params.has_rcs = !config_.is_drwig();
        return params;
    }

    // Effective camera matrix for an already-rectified image, per the rectify-before-fusion
    // convention shared with HomographyTab::estimate_camera_pose_pnp.
    cv::Matx33d effective_camera_matrix() const {
        const cv::Mat& m = intrinsics_projection_matrix_.empty() ? intrinsics_camera_matrix_ : intrinsics_projection_matrix_;
        return cv::Matx33d(m);
    }

    // ---------------------------------------------------------------------------------------
    // Live camera preview
    // ---------------------------------------------------------------------------------------

    void start_camera_preview() {
        if (camera_combo_->count() == 0) {
            status_label_->setText("No camera selected.");
            camera_preview_toggle_->blockSignals(true);
            camera_preview_toggle_->setChecked(false);
            camera_preview_toggle_->blockSignals(false);
            return;
        }
        const int idx = camera_combo_->currentData().toInt();
        cap_.open(idx, cv::CAP_V4L2);
        if (!cap_.isOpened()) cap_.open(idx, cv::CAP_ANY);
        if (!cap_.isOpened()) {
            status_label_->setText(QString("Failed to open camera %1.").arg(idx));
            camera_preview_toggle_->blockSignals(true);
            camera_preview_toggle_->setChecked(false);
            camera_preview_toggle_->blockSignals(false);
            return;
        }
        if (intrinsics_loaded_) {
            // MJPG is required to unlock the intrinsics' capture resolution on most UVC cameras --
            // see the same fix in HomographyTab/IntrinsicsTab's preview start. Without this, V4L2
            // silently falls back to its low default resolution (often 640x480).
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, intrinsics_image_size_.width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, intrinsics_image_size_.height);
        }
        if (!timer_->isActive()) timer_->start();
        set_toggle_button_visual(camera_preview_toggle_, true);
        apply_ui_state();
    }

    void stop_camera_preview() {
        if (cap_.isOpened()) cap_.release();
        if (!radar_reader_.is_running() && timer_->isActive()) timer_->stop();
        camera_preview_toggle_->blockSignals(true);
        camera_preview_toggle_->setChecked(false);
        camera_preview_toggle_->blockSignals(false);
        set_toggle_button_visual(camera_preview_toggle_, false);
        preview_label_->setText("Camera preview stopped.");
        apply_ui_state();
    }

    // ---------------------------------------------------------------------------------------
    // Live radar
    // ---------------------------------------------------------------------------------------

    void start_radar_live() {
        std::string error;
        const auto protocol = config_.is_drwig() ? bev::radarcam::RadarProtocol::kDrwigLrrPro
                                                 : bev::radarcam::RadarProtocol::kColumbusFc4d;
        if (!radar_reader_.start(can_interface_->text().toStdString(), error, protocol, config_.drwig_radar_id)) {
            radar_status_label_->setText(QString("Radar: failed to start (%1)").arg(QString::fromStdString(error)));
            radar_live_toggle_->blockSignals(true);
            radar_live_toggle_->setChecked(false);
            radar_live_toggle_->blockSignals(false);
            return;
        }
        if (!timer_->isActive()) timer_->start();
        set_toggle_button_visual(radar_live_toggle_, true);
        apply_ui_state();
    }

    void stop_radar_live() {
        radar_reader_.stop();
        if (!cap_.isOpened() && timer_->isActive()) timer_->stop();
        radar_live_toggle_->blockSignals(true);
        radar_live_toggle_->setChecked(false);
        radar_live_toggle_->blockSignals(false);
        set_toggle_button_visual(radar_live_toggle_, false);
        radar_status_label_->setText("Radar: not connected.");
        apply_ui_state();
    }

    // ---------------------------------------------------------------------------------------
    // Shared tick: camera frame grab/overlay + radar snapshot poll + capture accumulation
    // ---------------------------------------------------------------------------------------

    void on_frame_tick() {
        if (cap_.isOpened()) {
            cv::Mat frame;
            if (cap_.read(frame) && !frame.empty()) {
                cv::Mat rectified;
                if (intrinsics_loaded_) {
                    cv::undistort(frame, rectified, intrinsics_camera_matrix_, intrinsics_dist_coeffs_,
                        intrinsics_projection_matrix_);
                } else {
                    rectified = frame;
                }
                last_frame_clone_ = rectified.clone();

                cv::Mat overlay = rectified.clone();
                std::vector<cv::Point2f> corners;
                const cv::Size board_size(config_.board.inner_corners_x, config_.board.inner_corners_y);
                if (cv::findChessboardCorners(
                        overlay, board_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE)) {
                    cv::drawChessboardCorners(overlay, board_size, corners, true);
                }
                const QImage image = mat_to_qimage(overlay);
                if (!image.isNull()) {
                    preview_label_->setPixmap(
                        QPixmap::fromImage(image).scaled(preview_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
                }

                // Live radar-point overlay (bottom diagnostics pane): project every raw detection in
                // the current dwell snapshot into this frame with the extrinsic loaded by the "Live
                // Overlay" button (see start_live_overlay()), so alignment against a saved calibration
                // is visible in real time without running Solve in this session.
                //
                // Throttled to every 3rd tick (~10Hz) and downscaled BEFORE the BGR->RGB convert/copy/
                // Qt-scale in show_diagnostic_image -- doing that full pipeline at full camera
                // resolution on every 33ms tick (stacked on top of this block's own
                // findChessboardCorners call) is what made the overlay lag.
                if (live_overlay_enabled_ && radar_reader_.is_running() && (++live_overlay_tick_counter_ % 3 == 0)) {
                    cv::Mat radar_overlay = rectified.clone();
                    const cv::Matx33d K = effective_camera_matrix();
                    const auto radar_snap = radar_reader_.snapshot();
                    for (const auto& d : radar_snap.detections) {
                        const cv::Vec3d qv(d.position.x, d.position.y, d.position.z);
                        const cv::Vec3d p_cam = live_overlay_transform_.R * qv + live_overlay_transform_.t;
                        if (p_cam[2] <= 0.05) continue;  // behind or at the camera -- not projectable
                        const cv::Point2d px = bev::radarcam::project_pinhole(K, cv::Point3d(p_cam[0], p_cam[1], p_cam[2]));
                        if (px.x < 0 || px.y < 0 || px.x >= radar_overlay.cols || px.y >= radar_overlay.rows) continue;
                        cv::circle(radar_overlay, cv::Point(static_cast<int>(px.x), static_cast<int>(px.y)), 5,
                            cv::Scalar(38, 89, 217), 2, cv::LINE_AA);
                    }
                    cv::Mat radar_overlay_small;
                    cv::resize(radar_overlay, radar_overlay_small, cv::Size(), 0.4, 0.4, cv::INTER_AREA);
                    show_diagnostic_image(radar_overlay_small);
                }
            }
        }

        if (radar_reader_.is_running()) {
            const auto snap = radar_reader_.snapshot();
            radar_status_label_->setText(QString("Radar: connected. Detections this dwell: %1 (frame_id=%2, msgs=%3)")
                .arg(static_cast<int>(snap.detections.size()))
                .arg(snap.frame_id)
                .arg(snap.total_messages));

            // Gate on frame_id (increments once per completed radar dwell), not total_messages
            // (increments on every individual CAN-FD frame, including partial/in-progress dwells)
            // -- otherwise this would append many duplicate copies of the same stale dwell before
            // a genuinely new one finishes reassembling.
            if (capturing_ && snap.frame_id != last_captured_frame_id_) {
                last_captured_frame_id_ = snap.frame_id;
                capture_accumulator_.frames.push_back(snap.detections);
                --capture_ticks_remaining_;
                status_label_->setText(
                    QString("Capturing... %1 dwell frame(s) remaining.").arg(capture_ticks_remaining_));
                if (capture_ticks_remaining_ <= 0) {
                    finish_capture();
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Capture
    // ---------------------------------------------------------------------------------------

    void begin_capture() {
        if (!cap_.isOpened()) {
            status_label_->setText("Start Camera Preview before capturing.");
            return;
        }
        if (!radar_reader_.is_running()) {
            status_label_->setText("Start Radar Live before capturing.");
            return;
        }
        capturing_ = true;
        capture_accumulator_ = bev::radarcam::RadarDwell{};
        capture_ticks_remaining_ = 25;  // ~25 dwell publishes per capture, per spec's "20-50 frames per dwell"
        last_captured_frame_id_ = -1;
        status_label_->setText("Capturing...");
    }

    void finish_capture() {
        capturing_ = false;
        if (last_frame_clone_.empty()) {
            status_label_->setText("Capture aborted: no camera frame available.");
            return;
        }
        if (!intrinsics_loaded_) {
            status_label_->setText("Capture aborted: load intrinsics first.");
            return;
        }

        const auto board_result = bev::radarcam::detect_board_and_solve_pnp(last_frame_clone_,
            config_.board.inner_corners_x, config_.board.inner_corners_y, config_.board.square_size_m,
            effective_camera_matrix(), config_.x_b_measured_m, config_.pnp_reproj_rms_max_px, config_.board_min_tilt_deg);
        if (!board_result.accepted) {
            status_label_->setText(
                QString("Capture rejected: %1").arg(QString::fromStdString(board_result.reject_reason)));
            return;
        }

        bev::radarcam::Capture capture;
        const int idx = radar_cam_tab_detail::find_next_capture_index(captures_dir_->text().toStdString());
        capture.capture_id = QString("capture_%1").arg(idx, 4, 10, QChar('0')).toStdString();
        capture.image = last_frame_clone_;
        capture.radar_dwell = capture_accumulator_;
        capture.meta.timestamp = QDateTime::currentDateTimeUtc().toString(Qt::ISODate).toStdString();

        const std::string dir = captures_dir_->text().toStdString() + "/" + capture.capture_id;
        std::string error;
        if (!bev::radarcam::save_capture(dir, capture, error)) {
            status_label_->setText(QString("Capture save failed: %1").arg(QString::fromStdString(error)));
            return;
        }
        status_label_->setText(QString("Saved %1 (board tilt %2 deg, reproj RMS %3 px).")
            .arg(QString::fromStdString(capture.capture_id))
            .arg(board_result.pose.tilt_deg, 0, 'f', 1)
            .arg(board_result.pose.reprojection_rms_px, 0, 'f', 3));
    }

    // ---------------------------------------------------------------------------------------
    // Loading captures + board_pnp for Inspect/Solve/Validate
    // ---------------------------------------------------------------------------------------

    struct LoadedCapture {
        bev::radarcam::Capture capture;
        bev::radarcam::BoardDetectionResult board;
    };

    std::vector<LoadedCapture> load_all_captures_with_pnp(QStringList& warnings) {
        std::vector<LoadedCapture> out;
        for (const auto& dir : radar_cam_tab_detail::list_capture_dirs(captures_dir_->text().toStdString())) {
            bev::radarcam::Capture capture;
            std::string error;
            if (!bev::radarcam::load_capture(dir, capture, error)) {
                warnings << QString::fromStdString(dir + ": " + error);
                continue;
            }
            const auto board = bev::radarcam::detect_board_and_solve_pnp(capture.image, config_.board.inner_corners_x,
                config_.board.inner_corners_y, config_.board.square_size_m, effective_camera_matrix(),
                config_.x_b_measured_m, config_.pnp_reproj_rms_max_px, config_.board_min_tilt_deg);
            if (!board.accepted) {
                warnings << QString::fromStdString(dir + ": " + board.reject_reason);
                continue;
            }
            out.push_back({capture, board});
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------
    // §9.3 Inspect
    // ---------------------------------------------------------------------------------------

    void run_inspect() {
        if (!intrinsics_loaded_) {
            status_label_->setText("Load intrinsics before running Inspect.");
            return;
        }
        QStringList warnings;
        const auto loaded = load_all_captures_with_pnp(warnings);
        if (loaded.size() < 3) {
            status_label_->setText(QString("Inspect needs >=3 usable captures, found %1.").arg(static_cast<int>(loaded.size())));
            return;
        }

        const cv::Matx33d R_coarse = radar_cam_tab_detail::euler_deg_to_rotation(config_.coarse_euler_deg);
        const cv::Vec3d& t_coarse = config_.coarse_translation_m;

        bev::radarcam::ClutterFilterParams params = clutter_params();
        params.gate_radius_m = config_.gate_radii_m.empty() ? 2.0 : config_.gate_radii_m.front();

        std::vector<cv::Point3d> positions;
        for (const auto& lc : loaded) {
            const cv::Point3d q_hat = bev::radarcam::predict_radar_frame_position(lc.board.p_camera, R_coarse, t_coarse);
            const auto filtered = bev::radarcam::run_clutter_pipeline(
                lc.capture.radar_dwell, lc.capture.radar_background, lc.capture.has_background, q_hat, params);
            if (filtered.accepted) positions.push_back(filtered.q_radar);
        }
        if (positions.size() < 3) {
            status_label_->setText("Inspect: fewer than 3 captures survived the clutter filter with the coarse extrinsic.");
            return;
        }

        const bool planar = config_.is_drwig();
        const auto diag = bev::radarcam::compute_capture_distribution_diagnostic(
            positions, config_.svd_ratio_warn_threshold, 0.3, planar);
        std::vector<bool> all_inliers(positions.size(), true);
        show_diagnostic_image(bev::radarcam::render_reflector_cloud(positions, all_inliers));

        QString msg = QString("Inspect: %1 usable captures. Range spread %2m, azimuth spread %3deg")
            .arg(static_cast<int>(positions.size()))
            .arg(diag.range_spread_m, 0, 'f', 2)
            .arg(diag.azimuth_spread_deg, 0, 'f', 1);
        msg += planar ? QString(" (2D radar: height not measured, not needed).")
                      : QString(", height spread %1m.").arg(diag.height_spread_m, 0, 'f', 2);
        if (diag.degenerate_warning) {
            msg += QString(" WARNING: singular-value ratio %1 < %2 -- capture geometry may be near-degenerate.")
                .arg(diag.singular_value_ratio, 0, 'f', 3).arg(config_.svd_ratio_warn_threshold, 0, 'f', 2);
        }
        if (diag.height_spread_flagged) {
            msg += " WARNING: height spread is small -- vary the rig height across captures (§10).";
        }
        if (!warnings.isEmpty()) {
            msg += QString(" (%1 capture(s) skipped)").arg(warnings.size());
        }
        status_label_->setText(msg);
    }

    // ---------------------------------------------------------------------------------------
    // Solve
    // ---------------------------------------------------------------------------------------

    void run_solve() {
        if (!intrinsics_loaded_) {
            status_label_->setText("Load intrinsics before running Solve.");
            return;
        }
        if (config_.gate_radii_m.empty()) {
            status_label_->setText("Config has no gate_radii_m entries -- cannot run the iterate loop.");
            return;
        }

        QStringList warnings;
        const auto loaded = load_all_captures_with_pnp(warnings);
        if (loaded.size() < 3) {
            status_label_->setText(QString("Solve needs >=3 usable captures, found %1.").arg(static_cast<int>(loaded.size())));
            return;
        }

        std::vector<bev::radarcam::CaptureInput> inputs;
        for (const auto& lc : loaded) {
            bev::radarcam::CaptureInput ci;
            ci.p_camera = lc.board.p_camera;
            ci.board_pose = lc.board.pose;
            ci.dwell = lc.capture.radar_dwell;
            ci.background = lc.capture.radar_background;
            ci.has_background = lc.capture.has_background;
            inputs.push_back(ci);
        }

        const cv::Matx33d R_coarse = radar_cam_tab_detail::euler_deg_to_rotation(config_.coarse_euler_deg);
        const bool planar = config_.is_drwig();

        const auto iterate_result = bev::radarcam::run_gate_iterate(inputs, config_.gate_radii_m, R_coarse,
            config_.coarse_translation_m, clutter_params(), config_.ransac_inlier_threshold_m,
            config_.inlier_drop_warn_frac, planar);
        if (!iterate_result.success) {
            status_label_->setText("Solve failed: not enough inlier correspondences survived the gate-tightening loop.");
            show_diagnostic_image(bev::radarcam::render_inlier_vs_iteration(iterate_result.iterations));
            return;
        }

        std::vector<bev::radarcam::RefinementInput> refine_inputs;
        for (int idx : iterate_result.final_inlier_indices) {
            bev::radarcam::RefinementInput ri;
            ri.p_camera = inputs[idx].p_camera;
            ri.q_radar = iterate_result.per_capture[idx].q_radar;
            ri.board_pose = inputs[idx].board_pose;
            refine_inputs.push_back(ri);
        }

        bev::radarcam::RigidTransform final_transform = iterate_result.transform;
        cv::Vec3d x_b_estimated = config_.x_b_measured_m;
        // Planar: the closed-form planar RANSAC refit is already the least-squares answer for the
        // DOF a 2D radar observes. Ceres Mode A/B refine all 6 DOF (and Mode B the reflector offset),
        // which would let pitch/roll/height drift on data that can't constrain them -- skip.
        if (!planar && refine_inputs.size() >= 3) {
            if (config_.refine_mode == "B") {
                const auto mode_b = bev::radarcam::refine_mode_b(
                    refine_inputs, iterate_result.transform, config_.x_b_measured_m, config_.huber_delta_m);
                if (mode_b.success) {
                    final_transform = mode_b.transform;
                    x_b_estimated = mode_b.x_b_estimated;
                }
            } else {
                const auto mode_a = bev::radarcam::refine_mode_a(refine_inputs, iterate_result.transform, config_.huber_delta_m);
                if (mode_a.success) final_transform = mode_a.transform;
            }
            if (config_.run_mode_b_diagnostic && config_.refine_mode != "B") {
                const auto diag_b = bev::radarcam::refine_mode_b(
                    refine_inputs, iterate_result.transform, config_.x_b_measured_m, config_.huber_delta_m);
                if (diag_b.success) x_b_estimated = diag_b.x_b_estimated;
            }
        }

        last_solve_transform_ = final_transform;
        has_solve_ = true;

        bev::radarcam::ExtrinsicsData extrinsics;
        extrinsics.R = final_transform.R;
        extrinsics.t = final_transform.t;
        cv::Vec3d euler_deg_zyx;
        {
            cv::Mat mtxR, mtxQ;
            const cv::Vec3d angles_deg = cv::RQDecomp3x3(cv::Mat(final_transform.R), mtxR, mtxQ);
            euler_deg_zyx = angles_deg;
        }
        extrinsics.euler_deg = euler_deg_zyx;
        extrinsics.n_inliers = static_cast<int>(iterate_result.final_inlier_indices.size());
        extrinsics.n_total = static_cast<int>(inputs.size());
        extrinsics.x_b_measured_m = config_.x_b_measured_m;
        extrinsics.x_b_estimated_m = x_b_estimated;
        extrinsics.x_b_drift_m = cv::norm(x_b_estimated - config_.x_b_measured_m);
        extrinsics.funnel = radar_cam_tab_detail::aggregate_funnel(iterate_result.per_capture);

        double sq_sum = 0.0, sq_sum_px = 0.0;
        const cv::Matx33d K = effective_camera_matrix();
        for (int idx : iterate_result.final_inlier_indices) {
            cv::Point3d q = iterate_result.per_capture[idx].q_radar;
            if (planar) q = bev::radarcam::complete_planar_point(q, inputs[idx].p_camera, final_transform);
            const cv::Vec3d qv(q.x, q.y, q.z);
            const cv::Vec3d predicted = final_transform.R * qv + final_transform.t;
            const cv::Point3d predicted_pt(predicted[0], predicted[1], predicted[2]);
            const double r = cv::norm(predicted_pt - inputs[idx].p_camera);
            sq_sum += r * r;

            const cv::Point2d px_camera = bev::radarcam::project_pinhole(K, inputs[idx].p_camera);
            const cv::Point2d px_radar = bev::radarcam::project_pinhole(K, predicted_pt);
            const double rp = cv::norm(px_camera - px_radar);
            sq_sum_px += rp * rp;
        }
        const int n_inlier_pts = static_cast<int>(iterate_result.final_inlier_indices.size());
        extrinsics.rms_residual_m = n_inlier_pts == 0 ? 0.0 : std::sqrt(sq_sum / n_inlier_pts);
        extrinsics.rms_reprojection_px = n_inlier_pts == 0 ? 0.0 : std::sqrt(sq_sum_px / n_inlier_pts);

        std::string save_error;
        const bool saved = bev::radarcam::save_extrinsics_yaml(output_extrinsics_path_->text().toStdString(), extrinsics, save_error);

        show_diagnostic_image(bev::radarcam::render_inlier_vs_iteration(iterate_result.iterations));

        QString msg = planar
            ? QString("Solve (2D planar: yaw + in-plane offset; pitch/roll/height kept from config): %1/%2 inliers, "
                      "in-plane RMS %3 m.")
                  .arg(extrinsics.n_inliers)
                  .arg(extrinsics.n_total)
                  .arg(extrinsics.rms_residual_m, 0, 'f', 4)
            : QString("Solve: %1/%2 inliers, RMS %3 m, X_B drift %4 m.")
                  .arg(extrinsics.n_inliers)
                  .arg(extrinsics.n_total)
                  .arg(extrinsics.rms_residual_m, 0, 'f', 4)
                  .arg(extrinsics.x_b_drift_m, 0, 'f', 4);
        for (const auto& rec : iterate_result.iterations) {
            if (rec.guardrail_triggered) {
                msg += QString(" GUARDRAIL (iter %1): %2").arg(rec.iteration).arg(QString::fromStdString(rec.guardrail_message));
            }
        }
        msg += saved ? QString(" Saved to %1.").arg(output_extrinsics_path_->text())
                     : QString(" Save failed: %1.").arg(QString::fromStdString(save_error));
        status_label_->setText(msg);
        apply_ui_state();
    }

    // ---------------------------------------------------------------------------------------
    // Validate
    // ---------------------------------------------------------------------------------------

    void run_validate() {
        if (!has_solve_) {
            status_label_->setText("Run Solve before Validate.");
            return;
        }
        QStringList warnings;
        const auto loaded = load_all_captures_with_pnp(warnings);
        if (loaded.size() < 5) {
            status_label_->setText("Validate needs >=5 usable captures to hold any out.");
            return;
        }

        const cv::Matx33d R_coarse = radar_cam_tab_detail::euler_deg_to_rotation(config_.coarse_euler_deg);
        bev::radarcam::ClutterFilterParams params = clutter_params();
        params.gate_radius_m = config_.gate_radii_m.empty() ? 0.5 : config_.gate_radii_m.back();

        const int n_holdout = std::max(1, static_cast<int>(loaded.size() * config_.holdout_fraction));
        std::vector<bev::radarcam::HeldoutInput> heldout;
        std::vector<size_t> heldout_loaded_indices;  // index into `loaded`, parallel to `heldout` -- so the
                                                      // worst-residual overlay below can find its source image
        for (size_t i = loaded.size() - static_cast<size_t>(n_holdout); i < loaded.size(); ++i) {
            const auto& lc = loaded[i];
            const cv::Point3d q_hat =
                bev::radarcam::predict_radar_frame_position(lc.board.p_camera, last_solve_transform_.R, last_solve_transform_.t);
            const auto filtered =
                bev::radarcam::run_clutter_pipeline(lc.capture.radar_dwell, lc.capture.radar_background, lc.capture.has_background, q_hat, params);
            if (filtered.accepted) {
                bev::radarcam::HeldoutInput hi;
                hi.p_camera = lc.board.p_camera;
                // Planar: fill in the height the 2D radar can't measure, so the residuals below (and
                // the worst-residual overlay) report only the in-plane error the solve is accountable for.
                hi.q_radar = config_.is_drwig()
                    ? bev::radarcam::complete_planar_point(filtered.q_radar, lc.board.p_camera, last_solve_transform_)
                    : filtered.q_radar;
                heldout.push_back(hi);
                heldout_loaded_indices.push_back(i);
            }
        }
        if (heldout.empty()) {
            status_label_->setText("Validate: no held-out captures survived the clutter filter.");
            return;
        }

        const auto stats = bev::radarcam::compute_heldout_residuals(heldout, last_solve_transform_, effective_camera_matrix());

        // Overlay the worst-residual held-out capture on its actual image (render_reprojection_overlay,
        // diagnostics_render.hpp) so a bad RMS is visible directly, not just a number.
        const cv::Matx33d K = effective_camera_matrix();
        int worst_k = -1;
        double worst_residual_m = -1.0;
        for (size_t k = 0; k < heldout.size(); ++k) {
            const cv::Vec3d qv(heldout[k].q_radar.x, heldout[k].q_radar.y, heldout[k].q_radar.z);
            const cv::Vec3d predicted = last_solve_transform_.R * qv + last_solve_transform_.t;
            const double r = cv::norm(cv::Point3d(predicted[0], predicted[1], predicted[2]) - heldout[k].p_camera);
            if (r > worst_residual_m) {
                worst_residual_m = r;
                worst_k = static_cast<int>(k);
            }
        }
        QString overlay_note;
        if (worst_k >= 0) {
            const auto& worst_capture = loaded[heldout_loaded_indices[static_cast<size_t>(worst_k)]].capture;
            const cv::Vec3d qv(heldout[worst_k].q_radar.x, heldout[worst_k].q_radar.y, heldout[worst_k].q_radar.z);
            const cv::Vec3d predicted = last_solve_transform_.R * qv + last_solve_transform_.t;
            const cv::Point2d measured_px = bev::radarcam::project_pinhole(K, heldout[worst_k].p_camera);
            const cv::Point2d predicted_px =
                bev::radarcam::project_pinhole(K, cv::Point3d(predicted[0], predicted[1], predicted[2]));
            show_diagnostic_image(bev::radarcam::render_reprojection_overlay(worst_capture.image, measured_px, predicted_px));
            overlay_note = QString(" | Overlay: %1 (worst held-out, residual %2m).")
                               .arg(QString::fromStdString(worst_capture.capture_id))
                               .arg(worst_residual_m, 0, 'f', 3);
        }

        QString msg = QString("Validate (n=%1): %2 RMS %3m median %4m p95 %5m | reprojection RMS %6px median %7px p95 %8px.")
            .arg(stats.n_holdout)
            .arg(config_.is_drwig() ? "in-plane" : "3D")
            .arg(stats.rms_3d_m, 0, 'f', 4)
            .arg(stats.median_3d_m, 0, 'f', 4)
            .arg(stats.p95_3d_m, 0, 'f', 4)
            .arg(stats.rms_reprojection_px, 0, 'f', 2)
            .arg(stats.median_reprojection_px, 0, 'f', 2)
            .arg(stats.p95_reprojection_px, 0, 'f', 2);

        if (homography_loaded_) {
            const auto ground_check =
                bev::radarcam::ground_plane_cross_check(homography_data_, effective_camera_matrix(), last_solve_transform_);
            if (ground_check.success) {
                msg += QString(" | Ground check: derived radar height %1m, pitch %2deg, roll %3deg.")
                    .arg(ground_check.derived_radar_height_m, 0, 'f', 3)
                    .arg(ground_check.radar_pitch_deg, 0, 'f', 2)
                    .arg(ground_check.radar_roll_deg, 0, 'f', 2);
            } else {
                msg += QString(" | Ground check failed: %1.").arg(QString::fromStdString(ground_check.error));
            }
        } else {
            msg += " | No homography loaded -- §9.2 ground-plane cross-check skipped.";
        }
        msg += overlay_note;
        status_label_->setText(msg);
    }

    // ---------------------------------------------------------------------------------------
    // Shared UI helpers
    // ---------------------------------------------------------------------------------------

    void show_diagnostic_image(const cv::Mat& canvas) {
        const QImage image = mat_to_qimage(canvas);
        if (!image.isNull()) {
            diagnostics_label_->setPixmap(
                QPixmap::fromImage(image).scaled(diagnostics_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

    void apply_ui_state() {
        capture_btn_->setEnabled(cap_.isOpened() && radar_reader_.is_running() && !capturing_);
        inspect_btn_->setEnabled(intrinsics_loaded_);
        solve_btn_->setEnabled(intrinsics_loaded_ && !config_.gate_radii_m.empty());
        validate_btn_->setEnabled(has_solve_);
        live_overlay_toggle_->setEnabled(intrinsics_loaded_);
    }

    // ---------------------------------------------------------------------------------------
    // Live overlay: loads a saved extrinsics.yaml (independent of an in-session Solve) plus the
    // intrinsics/homography already in memory, then projects raw radar detections onto the live
    // preview every tick (see on_frame_tick) as a real-time calibration sanity check.
    // ---------------------------------------------------------------------------------------

    void start_live_overlay() {
        if (!intrinsics_loaded_) {
            status_label_->setText("Live Overlay needs intrinsics loaded first.");
            live_overlay_toggle_->blockSignals(true);
            live_overlay_toggle_->setChecked(false);
            live_overlay_toggle_->blockSignals(false);
            return;
        }
        load_homography();  // optional -- not required for the overlay math itself, only for parity
                             // with what Validate's ground-plane cross-check expects to have loaded

        bev::radarcam::ExtrinsicsData extrinsics;
        std::string error;
        if (!bev::radarcam::load_extrinsics_yaml(output_extrinsics_path_->text().toStdString(), extrinsics, error)) {
            status_label_->setText(QString("Live Overlay: failed to load extrinsics (%1)").arg(QString::fromStdString(error)));
            live_overlay_toggle_->blockSignals(true);
            live_overlay_toggle_->setChecked(false);
            live_overlay_toggle_->blockSignals(false);
            return;
        }
        live_overlay_transform_.R = extrinsics.R;
        live_overlay_transform_.t = extrinsics.t;
        live_overlay_enabled_ = true;
        set_toggle_button_visual(live_overlay_toggle_, true);
        status_label_->setText(QString("Live Overlay: loaded %1%2. Turn on Camera Preview + Radar Live to see it.")
            .arg(output_extrinsics_path_->text())
            .arg(homography_loaded_ ? " + homography" : " (no homography loaded)"));
    }

    void stop_live_overlay() {
        live_overlay_enabled_ = false;
        set_toggle_button_visual(live_overlay_toggle_, false);
        live_overlay_toggle_->blockSignals(true);
        live_overlay_toggle_->setChecked(false);
        live_overlay_toggle_->blockSignals(false);
    }

    // ---------------------------------------------------------------------------------------
    // Widgets
    // ---------------------------------------------------------------------------------------

    QComboBox* camera_combo_ = nullptr;
    QPushButton* refresh_btn_ = nullptr;
    QLineEdit* intrinsics_path_ = nullptr;
    QLineEdit* homography_path_ = nullptr;
    QLineEdit* config_path_ = nullptr;
    QLineEdit* can_interface_ = nullptr;
    QLineEdit* captures_dir_ = nullptr;
    QLineEdit* output_extrinsics_path_ = nullptr;
    QToolButton* camera_preview_toggle_ = nullptr;
    QToolButton* radar_live_toggle_ = nullptr;
    QToolButton* live_overlay_toggle_ = nullptr;
    QPushButton* capture_btn_ = nullptr;
    QPushButton* inspect_btn_ = nullptr;
    QPushButton* solve_btn_ = nullptr;
    QPushButton* validate_btn_ = nullptr;
    QLabel* preview_label_ = nullptr;
    QLabel* diagnostics_label_ = nullptr;
    QLabel* intrinsics_status_label_ = nullptr;
    QLabel* config_status_label_ = nullptr;
    QLabel* radar_status_label_ = nullptr;
    QLabel* status_label_ = nullptr;
    QTimer* timer_ = nullptr;

    // ---------------------------------------------------------------------------------------
    // State
    // ---------------------------------------------------------------------------------------

    cv::VideoCapture cap_;
    cv::Size intrinsics_image_size_;
    cv::Mat intrinsics_camera_matrix_;
    cv::Mat intrinsics_dist_coeffs_;
    cv::Mat intrinsics_projection_matrix_;
    bool intrinsics_loaded_ = false;

    bev::HomographyData homography_data_;
    bool homography_loaded_ = false;

    bev::radarcam::RadarCamConfig config_;

    bev::radarcam::SocketCanReader radar_reader_;

    cv::Mat last_frame_clone_;
    bool capturing_ = false;
    int capture_ticks_remaining_ = 0;
    int last_captured_frame_id_ = -1;
    bev::radarcam::RadarDwell capture_accumulator_;

    bev::radarcam::RigidTransform last_solve_transform_;
    bool has_solve_ = false;

    bev::radarcam::RigidTransform live_overlay_transform_;  // loaded from output_extrinsics_path_ by
                                                             // start_live_overlay(), independent of has_solve_
    bool live_overlay_enabled_ = false;
    int live_overlay_tick_counter_ = 0;  // throttles the redraw to every 3rd tick, see on_frame_tick
};
