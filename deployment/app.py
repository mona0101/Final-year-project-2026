import streamlit as st
import torch
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import tempfile
import imageio
import numpy as np

# Import your existing models
from all_models import LogMelCNN_64, Visual_MobileNet, RF_ResNet18
from fusion_model import LateFusionModel
from data_loader import (
    load_uploaded_data, cleanup_temp_dir, get_audio_id,
    get_audio_segments, get_segment_data,
    preprocess_audio, preprocess_video_frames, preprocess_rf_image,
    preprocess_all_segments
)

# ============================================================
# Helper functions to create MP4 videos from frames
# ============================================================
def create_mp4_from_frames(frame_paths, output_path, fps=30):
    """Create an MP4 video from image frames using imageio/ffmpeg."""
    if not frame_paths:
        return None

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        macro_block_size=1
    )

    try:
        for frame_path in frame_paths:
            if os.path.exists(frame_path):

                img = Image.open(frame_path).convert("RGB")

                # =====================================
                # FIX: Make width/height divisible by 2
                # =====================================
                width, height = img.size

                if width % 2 != 0:
                    width -= 1

                if height % 2 != 0:
                    height -= 1

                img = img.resize((width, height))

                writer.append_data(np.array(img))

    finally:
        writer.close()

    return output_path


def create_yolo_tracking_video(yolo_model, frame_paths, output_path, fps=30, conf_threshold=0.25):
    """Create an MP4 video where YOLO bounding boxes are drawn on every frame."""
    if not frame_paths:
        return None

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        format="FFMPEG",
        codec="libx264",
        macro_block_size=1
    )

    try:
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                continue

            results = yolo_model(frame_path, conf=conf_threshold, verbose=False)
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                plotted = result.plot()

                # Ultralytics returns BGR; convert to RGB for correct colors.
                if plotted.ndim == 3 and plotted.shape[-1] == 3:
                    plotted = plotted[:, :, ::-1]

                frame = plotted
            else:
                frame = np.array(Image.open(frame_path).convert("RGB"))

            writer.append_data(frame)
    finally:
        writer.close()

    return output_path


# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="Drone Detection", layout="wide")
st.title("Drone Detection System")
st.markdown("### Upload your data as ZIP files")


# ============================================================
# Load Fusion Model
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_backbone = LogMelCNN_64()
    video_backbone = Visual_MobileNet(n_classes=2)
    rf_backbone = RF_ResNet18(n_classes=2)

    model = LateFusionModel(
        model_audio=audio_backbone,
        model_visual=video_backbone,
        model_rf=rf_backbone
    )

    weights_path = os.path.join(
        os.path.dirname(__file__),
        "weights",
        "LogMel_Mobile_ResNet_weights (2).pth"
    )

    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    return model, device


# ============================================================
# Load YOLO Model
# ============================================================
@st.cache_resource
def load_yolo_model():
    yolo_path = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
    return YOLO(yolo_path)


with st.spinner("Loading fusion model..."):
    model, device = load_model()

with st.spinner("Loading YOLO model..."):
    yolo_model = load_yolo_model()

st.success(f"Models loaded successfully on {device}")


# ============================================================
# YOLO Detection Helper
# ============================================================
def run_yolo_detection(frame_paths, conf_threshold=0.25, max_preview_frames=6):
    """
    Runs YOLO on a list of video frames.
    Returns:
    - best detection confidence
    - best annotated frame
    - several consecutive annotated frames for preview
    """
    if not frame_paths:
        return {
            "detected": False,
            "confidence": 0.0,
            "image": None,
            "preview_images": [],
            "detected_frames": 0
        }

    best_conf = 0.0
    best_image = None
    preview_images = []
    detected_frames = 0

    for frame_path in frame_paths:
        if not os.path.exists(frame_path):
            continue

        results = yolo_model(frame_path, conf=conf_threshold, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            detected_frames += 1
            confs = result.boxes.conf.cpu().numpy()
            current_conf = float(np.max(confs))

            plotted = result.plot()

            # Ultralytics returns BGR; convert to RGB without OpenCV.
            if plotted.ndim == 3 and plotted.shape[-1] == 3:
                plotted = plotted[:, :, ::-1]

            annotated_image = Image.fromarray(plotted)

            if len(preview_images) < max_preview_frames:
                preview_images.append({
                    "image": annotated_image,
                    "confidence": current_conf,
                    "frame_name": os.path.basename(frame_path)
                })

            if current_conf > best_conf:
                best_conf = current_conf
                best_image = annotated_image

    return {
        "detected": best_conf >= conf_threshold,
        "confidence": best_conf,
        "image": best_image,
        "preview_images": preview_images,
        "detected_frames": detected_frames
    }


# ============================================================
# File Upload
# ============================================================
st.sidebar.header("Upload Data (ZIP files)")

audio_zip = st.sidebar.file_uploader("Audio ZIP (.wav files)", type=["zip"])
video_zip = st.sidebar.file_uploader("Video Frames ZIP (.jpg files)", type=["zip"])
rf_zip = st.sidebar.file_uploader("RF Spectrograms ZIP (.jpg files)", type=["zip"])

temp_dir = None
video_temp_path = None
rf_temp_path = None
yolo_tracking_video_path = None

if audio_zip and video_zip and rf_zip:
    with st.spinner("Extracting files..."):
        audio_files, video_files, rf_files, temp_dir = load_uploaded_data(
            audio_zip, video_zip, rf_zip
        )

    st.sidebar.success(f"Audio files: {len(audio_files)}")
    st.sidebar.success(f"Video frames: {len(video_files)}")
    st.sidebar.success(f"RF frames: {len(rf_files)}")

    # ============================================================
    # Select Audio File
    # ============================================================
    if audio_files:
        audio_ids = [get_audio_id(f) for f in audio_files]

        selected_audio_id = st.sidebar.selectbox(
            "Select Audio File",
            options=audio_ids,
            format_func=lambda x: f"{x}.wav"
        )

        selected_audio_path = [
            f for f in audio_files if get_audio_id(f) == selected_audio_id
        ][0]

        matching_video, matching_rf, num_segments = get_audio_segments(
            selected_audio_id, video_files, rf_files
        )

        if num_segments > 0:
            # ============================================================
            # Create MP4 videos
            # ============================================================
            with st.spinner("Creating videos from frames..."):
                if len(matching_video) >= 280:
                    video_temp_path = tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    ).name
                    create_mp4_from_frames(matching_video[:300], video_temp_path, fps=30)

                if len(matching_rf) >= 40:
                    rf_temp_path = tempfile.NamedTemporaryFile(
                        suffix=".mp4", delete=False
                    ).name
                    create_mp4_from_frames(matching_rf[:40], rf_temp_path, fps=4)

            # ============================================================
            # Display All Three Media
            # ============================================================
            st.markdown("## Full Media Preview")
            st.caption("Each uploaded sample represents approximately 10 seconds.")

            col_audio, col_video, col_rf = st.columns(3)

            with col_audio:
                st.markdown("### Audio")
                st.audio(selected_audio_path)
                st.caption("10-second audio file")

            with col_video:
                st.markdown("### Video Frames")
                if video_temp_path and os.path.exists(video_temp_path):
                    with open(video_temp_path, "rb") as video_file:
                        st.video(video_file.read())
                    st.caption("300 frames at 30 fps = 10 seconds")
                else:
                    st.warning(f"Only {len(matching_video)} video frames found")

            with col_rf:
                st.markdown("### RF Spectrograms")
                if rf_temp_path and os.path.exists(rf_temp_path):
                    with open(rf_temp_path, "rb") as rf_file:
                        st.video(rf_file.read())
                    st.caption("40 RF frames at 4 fps = 10 seconds")
                else:
                    st.warning(f"Only {len(matching_rf)} RF frames found")

            # ============================================================
            # Segment Selection
            # ============================================================
            st.divider()
            st.markdown("## Segment Analysis")
            st.caption("Each segment represents 0.25 seconds.")

            segment_index = st.slider(
                "Select Segment to Analyze",
                0,
                num_segments - 1,
                0,
                format="Segment %d"
            )

            video_segment, rf_frame = get_segment_data(
                matching_video, matching_rf, segment_index
            )

            start_time = segment_index * 0.25
            end_time = (segment_index + 1) * 0.25

            st.markdown(f"**Segment {segment_index}: {start_time:.2f}s – {end_time:.2f}s**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Audio Segment**")
                st.audio(selected_audio_path)

            with col2:
                st.markdown(f"**Video Frames ({segment_index * 7 + 1}–{segment_index * 7 + 7})**")
                if video_segment:
                    frame_cols = st.columns(7)
                    for i, (col, path) in enumerate(zip(frame_cols, video_segment)):
                        if os.path.exists(path):
                            img = Image.open(path)
                            col.image(img, caption=f"{i + 1}")

            with col3:
                st.markdown(f"**RF Frame ({segment_index + 1})**")
                if rf_frame and os.path.exists(rf_frame):
                    img = Image.open(rf_frame)
                    st.image(img, caption="RF Frame")

            st.divider()
            generate_yolo_video = st.checkbox(
                "Generate YOLO tracking video for the full 10-second video",
                value=False,
                help="This may take longer because YOLO will annotate many frames."
            )

            # ============================================================
            # Single-Segment Detection
            # ============================================================
            if st.button("Run Detection for Selected Segment", type="primary"):
                with st.spinner("Processing selected segment..."):
                    audio_tensor = preprocess_audio(
                        selected_audio_path,
                        segment_index=segment_index
                    )
                    video_tensor = preprocess_video_frames(video_segment)
                    rf_tensor = preprocess_rf_image(rf_frame) if rf_frame else None

                    audio_input = audio_tensor.unsqueeze(0).to(device)
                    video_input = video_tensor.unsqueeze(0).to(device) if video_tensor is not None else None
                    rf_input = rf_tensor.unsqueeze(0).to(device) if rf_tensor is not None else None

                    # YOLO visual detection on the selected video frames
                    yolo_result = run_yolo_detection(video_segment)
                    yolo_prob = yolo_result["confidence"]
                    yolo_pred = 1 if yolo_prob > 0.5 else 0

                    yolo_tracking_video_path = None
                    if generate_yolo_video:
                        yolo_tracking_video_path = tempfile.NamedTemporaryFile(
                            suffix=".mp4",
                            delete=False
                        ).name
                        create_yolo_tracking_video(
                            yolo_model,
                            matching_video[:300],
                            yolo_tracking_video_path,
                            fps=30
                        )

                    with torch.no_grad():
                        result = model(
                            audio=audio_input,
                            video=video_input,
                            rf=rf_input,
                            return_individual=True
                        )

                    fusion_prob = result["fusion"].item()
                    fusion_pred = 1 if fusion_prob > 0.5 else 0

                # ============================================================
                # Display Results
                # ============================================================
                st.divider()
                st.subheader("Detection Results")

                col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)

                with col_r1:
                    ap = result["individual"].get("audio")
                    ap = ap.item() if ap is not None else None
                    st.metric(
                        "Audio",
                        "Drone" if ap is not None and ap > 0.5 else "No Drone",
                        f"{ap:.2%}" if ap is not None else "N/A"
                    )

                with col_r2:
                    vp = result["individual"].get("video")
                    vp = vp.item() if vp is not None else None
                    st.metric(
                        "Video Model",
                        "Drone" if vp is not None and vp > 0.5 else "No Drone",
                        f"{vp:.2%}" if vp is not None else "N/A"
                    )

                with col_r3:
                    rp = result["individual"].get("rf")
                    rp = rp.item() if rp is not None else None
                    st.metric(
                        "RF",
                        "Drone" if rp is not None and rp > 0.5 else "No Drone",
                        f"{rp:.2%}" if rp is not None else "N/A"
                    )

                with col_r4:
                    st.metric(
                        "Fusion",
                        "Drone" if fusion_pred == 1 else "No Drone",
                        f"{fusion_prob:.2%}"
                    )

                with col_r5:
                    st.metric(
                        "YOLO",
                        "Drone" if yolo_pred == 1 else "No Drone",
                        f"{yolo_prob:.2%}"
                    )

                st.divider()

                if fusion_pred == 1:
                    st.error(f"Drone detected by fusion model. Confidence: {fusion_prob:.2%}")
                else:
                    st.success(f"No drone detected by fusion model. Confidence: {1 - fusion_prob:.2%}")

                st.caption(f"Modalities used by fusion model: {', '.join(result['modalities_used'])}")

                # ============================================================
                # YOLO Detection / Tracking Preview
                # ============================================================
                st.subheader("YOLO Detection / Tracking Preview")

                if yolo_result["image"] is not None:
                    yolo_summary_cols = st.columns(3)
                    yolo_summary_cols[0].metric("YOLO Confidence", f"{yolo_prob:.2%}")
                    yolo_summary_cols[1].metric("Detected Frames", yolo_result["detected_frames"])
                    yolo_summary_cols[2].metric("Preview Frames", len(yolo_result["preview_images"]))

                    st.markdown("#### Consecutive frames with bounding boxes")
                    preview_cols = st.columns(len(yolo_result["preview_images"]))

                    for col, preview in zip(preview_cols, yolo_result["preview_images"]):
                        col.image(
                            preview["image"],
                            caption=f"{preview['confidence']:.2%}",
                            use_container_width=True
                        )

                    st.markdown("#### Best YOLO detection frame")
                    st.image(
                        yolo_result["image"],
                        caption=f"Best detected frame | Confidence: {yolo_prob:.2%}",
                        use_container_width=True
                    )

                    if yolo_tracking_video_path and os.path.exists(yolo_tracking_video_path):
                        st.markdown("#### YOLO annotated tracking video")
                        with open(yolo_tracking_video_path, "rb") as yolo_video_file:
                            st.video(yolo_video_file.read())
                        st.caption("Bounding boxes are drawn frame-by-frame across the full video preview.")
                else:
                    st.info("YOLO did not detect a drone in this selected segment.")

            # ============================================================
            # Run ALL Segments
            # ============================================================
            st.divider()

            if st.button("Run All Segments", type="secondary"):
                with st.spinner(f"Processing all {num_segments} segments..."):
                    all_segments = preprocess_all_segments(
                        selected_audio_path,
                        matching_video,
                        matching_rf,
                        num_segments
                    )

                    results_rows = []
                    progress_bar = st.progress(0)

                    for i, seg in enumerate(all_segments):
                        audio_in = seg["audio"].unsqueeze(0).to(device)
                        video_in = seg["video"].unsqueeze(0).to(device) if seg["video"] is not None else None
                        rf_in = seg["rf"].unsqueeze(0).to(device) if seg["rf"] is not None else None

                        with torch.no_grad():
                            result = model(
                                audio=audio_in,
                                video=video_in,
                                rf=rf_in,
                                return_individual=True
                            )

                        ap = result["individual"].get("audio")
                        vp = result["individual"].get("video")
                        rp = result["individual"].get("rf")
                        fp = result["fusion"]

                        ap = ap.item() if ap is not None else None
                        vp = vp.item() if vp is not None else None
                        rp = rp.item() if rp is not None else None
                        fp = fp.item() if fp is not None else None

                        results_rows.append({
                            "Segment": i,
                            "Time": f"{i * 0.25:.2f}s – {(i + 1) * 0.25:.2f}s",
                            "Video frames": f"{i * 7 + 1}–{i * 7 + 7}",
                            "RF frame": str(i + 1),
                            "Audio prob": ap,
                            "Video prob": vp,
                            "RF prob": rp,
                            "Fusion prob": fp,
                            "Audio": "Drone" if ap is not None and ap > 0.5 else "Clear",
                            "Video": "Drone" if vp is not None and vp > 0.5 else "Clear",
                            "RF": "Drone" if rp is not None and rp > 0.5 else "Clear",
                            "Fusion": "Drone" if fp is not None and fp > 0.5 else "Clear",
                        })

                        progress_bar.progress((i + 1) / num_segments)

                    progress_bar.empty()
                    df = pd.DataFrame(results_rows)

                # Summary metrics
                st.subheader("Summary")
                drone_segs = int((df["Fusion prob"] > 0.5).sum())
                detection_pct = drone_segs / num_segments

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total segments", num_segments)
                c2.metric("Drone detected", drone_segs)
                c3.metric("No drone", num_segments - drone_segs)
                c4.metric("Detection rate", f"{detection_pct:.1%}")

                if drone_segs > 0:
                    st.error(
                        f"Drone detected in {drone_segs} of {num_segments} segments "
                        f"({detection_pct:.1%})."
                    )
                else:
                    st.success(f"No drone detected across all {num_segments} segments.")

                # Color-coded results table
                st.subheader("Per-Segment Results")

                def color_prob(val):
                    if val is None or not isinstance(val, float):
                        return "background-color: #888888; color: white"
                    r = int(220 * val)
                    g = int(200 * (1 - val))
                    return f"background-color: rgb({r},{g},60); color: white; font-weight: 500"

                display_df = df[[
                    "Segment", "Time", "Video frames", "RF frame",
                    "Audio", "Video", "RF", "Fusion",
                    "Audio prob", "Video prob", "RF prob", "Fusion prob"
                ]]

                styled = (
                    display_df.style
                    .applymap(
                        color_prob,
                        subset=["Audio prob", "Video prob", "RF prob", "Fusion prob"]
                    )
                    .format(
                        {
                            "Audio prob": "{:.2%}",
                            "Video prob": "{:.2%}",
                            "RF prob": "{:.2%}",
                            "Fusion prob": "{:.2%}"
                        },
                        na_rep="N/A"
                    )
                )

                st.dataframe(styled, use_container_width=True, height=600)

        else:
            st.sidebar.error(
                f"Need at least 7 video frames and 1 RF frame. "
                f"Found: {len(matching_video)} video, {len(matching_rf)} RF"
            )
    else:
        st.sidebar.error("No audio files found in the uploaded ZIP")
else:
    st.info("Upload audio.zip, video.zip, and rf.zip from the sidebar to start analysis.")


# ============================================================
# Cleanup
# ============================================================
if temp_dir:
    cleanup_temp_dir(temp_dir)

if video_temp_path and os.path.exists(video_temp_path):
    os.unlink(video_temp_path)

if rf_temp_path and os.path.exists(rf_temp_path):
    os.unlink(rf_temp_path)

if yolo_tracking_video_path and os.path.exists(yolo_tracking_video_path):
    os.unlink(yolo_tracking_video_path)

st.divider()
st.caption(
    "Drone Detection System | 10-second analysis | 30fps video | 4fps RF | Each segment = 0.25s"
)
