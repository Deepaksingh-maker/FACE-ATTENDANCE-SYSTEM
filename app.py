"""
app.py
Streamlit Face Recognition Attendance System.
Run with: py -3 -m streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import date
import io

import database as db
import face_utils as fu
import importlib
importlib.reload(db)
importlib.reload(fu)

ENGINE_TYPE = "dlib (Full AI)" if getattr(fu, "USE_DLIB", False) else "OpenCV Vision Engine"

st.set_page_config(
    page_title="FaceAttendance AI",
    page_icon=":material/center_focus_strong:",
    layout="wide",
)

db.init_db()


def image_to_np(uploaded_or_camera_file):
    img = Image.open(uploaded_or_camera_file).convert("RGB")
    return np.array(img)


def image_to_bytes(uploaded_or_camera_file):
    uploaded_or_camera_file.seek(0)
    return uploaded_or_camera_file.read()


# ---------------- Header & System Status ----------------

with st.container(border=True):
    col_header, col_badge = st.columns([3, 1], vertical_alignment="center")
    with col_header:
        st.title("FaceAttendance AI", text_alignment="left")
        st.caption("Next-generation facial recognition attendance management system")
    with col_badge:
        st.badge(f"{ENGINE_TYPE} Ready", icon=":material/check_circle:", color="green")
        st.caption("Face detection & matching active")



# ---------------- Sidebar Navigation ----------------

batches = db.get_batches()
batch_names = [b[1] for b in batches]
batch_lookup = {b[1]: b[0] for b in batches}  # name -> id

if hasattr(db, "get_stats_summary"):
    stats = db.get_stats_summary()
else:
    importlib.reload(db)
    stats = db.get_stats_summary()


with st.sidebar:
    st.markdown("### :material/explore: Menu")
    page = st.radio(
        "Navigation",
        ["Dashboard & Batches", "Register Student", "Take Attendance", "Attendance Reports"],
        format_func=lambda x: {
            "Dashboard & Batches": "📊  Dashboard & Batches",
            "Register Student": "👤  Register Student",
            "Take Attendance": "📸  Take Attendance",
            "Attendance Reports": "📈  Attendance Reports",
        }[x],
        label_visibility="collapsed",
    )

    st.space("medium")
    with st.container(border=True):
        st.markdown("**:material/insights: Quick System Stats**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Batches", stats["batches"])
        with col_s2:
            st.metric("Students", stats["students"])
        
        today_rate = (
            f"{(stats['today_present'] / stats['today_total'] * 100):.1f}%"
            if stats["today_total"] > 0
            else "N/A"
        )
        st.caption(f"Today's Attendance Rate: **{today_rate}**")


# ---------------- Dialog Modals ----------------

@st.dialog("Delete Batch")
def confirm_delete_batch(batch_id, batch_name):
    st.markdown(f"Are you sure you want to delete **{batch_name}**?")
    st.warning("⚠️ This will permanently remove all student records and attendance history associated with this batch.")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    with col_d2:
        if st.button("Confirm Delete", type="primary", icon=":material/delete:", width="stretch"):
            ok, msg = db.delete_batch(batch_id)
            if ok:
                st.toast(f"Deleted batch {batch_name}", icon="🗑️")
                st.rerun()


@st.dialog("Remove Student")
def confirm_delete_student(student_id, student_name):
    st.markdown(f"Are you sure you want to remove **{student_name}**?")
    st.caption("This student's past attendance records will also be removed.")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    with col_d2:
        if st.button("Remove Student", type="primary", icon=":material/delete:", width="stretch"):
            db.delete_student(student_id)
            st.toast(f"Removed student {student_name}", icon="🗑️")
            st.rerun()


# ---------------- Page: Dashboard & Batches ----------------

if page == "Dashboard & Batches":
    st.header("Dashboard & Batch Management", text_alignment="left")
    st.caption("Overview of registered batches, student distribution, and system configuration.")

    st.space("small")

    # KPI Summary Row
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        with st.container(border=True):
            st.metric("Total Batches", stats["batches"], help="Active academic or training batches", border=False)
    with kpi2:
        with st.container(border=True):
            st.metric("Enrolled Students", stats["students"], help="Total registered students across all batches", border=False)
    with kpi3:
        with st.container(border=True):
            att_text = f"{stats['today_present']} / {stats['today_total']}" if stats['today_total'] > 0 else "0 Records"
            st.metric("Today's Attendance", att_text, help="Attendance records marked today", border=False)

    st.space("medium")

    col_list, col_add = st.columns([3, 2])

    with col_list:
        with st.container(border=True):
            st.subheader(":material/school: Existing Batches", text_alignment="left")
            if batches:
                # Calculate student counts per batch
                batch_data = []
                for b_id, b_name in batches:
                    st_count = len(db.get_students_by_batch(b_id))
                    batch_data.append({"Batch ID": b_id, "Batch Name": b_name, "Students Enrolled": st_count})
                
                df_batches = pd.DataFrame(batch_data)
                st.dataframe(
                    df_batches,
                    column_config={
                        "Batch ID": st.column_config.NumberColumn("ID", width="small"),
                        "Batch Name": st.column_config.TextColumn("Batch Name", width="large"),
                        "Students Enrolled": st.column_config.NumberColumn("Enrolled Students", format="%d 👤"),
                    },
                    hide_index=True,
                    width="stretch",
                )

                st.space("small")
                st.markdown("**:material/settings: Batch Actions**")
                col_sel, col_del = st.columns([3, 1], vertical_alignment="bottom")
                with col_sel:
                    selected_del_name = st.selectbox(
                        "Select batch to delete",
                        batch_names,
                        key="del_batch_sel",
                        label_visibility="collapsed",
                    )
                with col_del:
                    if st.button("Delete Batch", icon=":material/delete:", type="secondary", width="stretch"):
                        confirm_delete_batch(batch_lookup[selected_del_name], selected_del_name)
            else:
                st.info("No batches created yet. Add your first batch on the right!")

    with col_add:
        with st.container(border=True):
            st.subheader(":material/add_circle: Create New Batch", text_alignment="left")
            with st.form("add_batch_form", border=False):
                new_batch = st.text_input(
                    "Batch Name",
                    placeholder="e.g. Data Science 2026, Web Dev Cohort 1",
                    help="Enter a unique descriptive name for the batch",
                )
                submitted = st.form_submit_button(
                    "Create Batch",
                    icon=":material/add_circle:",
                    type="primary",
                    width="stretch",
                )
                if submitted:
                    if new_batch.strip():
                        ok, msg = db.add_batch(new_batch.strip())
                        if ok:
                            st.toast(f"Batch '{new_batch.strip()}' created successfully!", icon="✅")
                            st.rerun()
                        else:
                            st.error(msg, icon=":material/error:")
                    else:
                        st.warning("Please enter a valid batch name.", icon=":material/warning:")


# ---------------- Page: Register Student ----------------

elif page == "Register Student":
    st.header("Student Registration", text_alignment="left")
    st.caption("Enroll new students by capturing front-facing facial biometric profiles.")

    if not batch_names:
        st.warning("Please create a batch first under **Dashboard & Batches** before enrolling students.", icon=":material/warning:")
        st.stop()

    st.space("small")

    with st.container(border=True):
        col_info, col_photo = st.columns([1, 1], gap="medium")

        with col_info:
            st.subheader(":material/badge: Student Profile", text_alignment="left")
            selected_batch = st.selectbox("Target Batch", batch_names, help="Select student's batch")
            name = st.text_input("Full Name", placeholder="e.g. John Doe")
            roll_no = st.text_input("Roll Number / Student ID", placeholder="e.g. CS2026-001")

        with col_photo:
            st.subheader(":material/add_a_photo: Biometric Photo Capture", text_alignment="left")
            capture_mode = st.segmented_control(
                "Photo Source",
                ["Camera", "Upload File"],
                default="Camera",
            )

            photo_file = None
            if capture_mode == "Camera":
                photo_file = st.camera_input("Capture front-facing portrait photo")
            else:
                photo_file = st.file_uploader(
                    "Upload high-resolution photo",
                    type=["jpg", "jpeg", "png"],
                    help="Upload a clear front-facing portrait",
                )

    if photo_file:
        st.space("small")
        with st.container(border=True):
            col_prev, col_act = st.columns([1, 2], vertical_alignment="center")
            with col_prev:
                st.image(photo_file, caption="Captured Portrait Preview", width=180)
            with col_act:
                st.markdown("### Ready to Register")
                st.caption("Ensure the student's face is clearly lit and unobscured.")
                if st.button("Submit & Register Student", icon=":material/person_add:", type="primary", width="stretch"):
                    if not name.strip() or not roll_no.strip():
                        st.error("Please fill in both student name and roll number.", icon=":material/error:")
                    else:
                        with st.spinner("Analyzing face biometrics..."):
                            img_np = image_to_np(photo_file)
                            encoding, err = fu.get_single_face_encoding(img_np)
                            if err:
                                st.error(err, icon=":material/error:")
                            else:
                                ok, msg = db.add_student(
                                    name.strip(),
                                    roll_no.strip(),
                                    batch_lookup[selected_batch],
                                    encoding,
                                    image_to_bytes(photo_file),
                                )
                                if ok:
                                    st.toast(f"Student '{name.strip()}' registered successfully!", icon="🎉")
                                    st.rerun()
                                else:
                                    st.error(msg, icon=":material/error:")

    st.space("medium")

    # Registered Students Roster & Deletion
    with st.container(border=True):
        st.subheader(":material/group: Batch Roster Management", text_alignment="left")
        curr_batch_id = batch_lookup[selected_batch]
        registered = db.get_students_by_batch(curr_batch_id)

        if registered:
            st.caption(f"Currently **{len(registered)}** student(s) enrolled in **{selected_batch}**")
            
            roster_df = pd.DataFrame(
                [{"ID": s["id"], "Name": s["name"], "Roll Number": s["roll_no"]} for s in registered]
            )
            st.dataframe(
                roster_df,
                column_config={
                    "ID": st.column_config.NumberColumn("ID", width="small"),
                    "Name": st.column_config.TextColumn("Student Name", width="large"),
                    "Roll Number": st.column_config.TextColumn("Roll No / ID", width="medium"),
                },
                hide_index=True,
                width="stretch",
            )

            st.space("small")
            col_rem_sel, col_rem_btn = st.columns([3, 1], vertical_alignment="bottom")
            with col_rem_sel:
                rem_map = {f"{s['name']} ({s['roll_no']})": (s["id"], s["name"]) for s in registered}
                selected_rem_label = st.selectbox(
                    "Select student to remove",
                    list(rem_map.keys()),
                    label_visibility="collapsed",
                )
            with col_rem_btn:
                if st.button("Remove Student", icon=":material/delete:", type="secondary", width="stretch"):
                    s_id, s_name = rem_map[selected_rem_label]
                    confirm_delete_student(s_id, s_name)
        else:
            st.info(f"No students registered in **{selected_batch}** yet.")


# ---------------- Page: Take Attendance ----------------

elif page == "Take Attendance":
    st.header("Mark Attendance", text_alignment="left")
    st.caption("Capture class photo to automatically detect and log present students using facial recognition.")

    if not batch_names:
        st.warning("Please create a batch first under **Dashboard & Batches**.", icon=":material/warning:")
        st.stop()

    st.space("small")

    # Workflow Guidance Cards
    g1, g2, g3 = st.columns(3)
    with g1:
        with st.container(border=True):
            st.markdown("**:material/looks_one: Select Session**")
            st.caption("Choose target batch & lecture date")
    with g2:
        with st.container(border=True):
            st.markdown("**:material/looks_two: Capture Image**")
            st.caption("Upload or snapshot group photo")
    with g3:
        with st.container(border=True):
            st.markdown("**:material/looks_3: AI Matching**")
            st.caption("Instant face detection & log")

    st.space("small")

    with st.container(border=True):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            selected_batch = st.selectbox("Select Batch", batch_names)
            batch_id = batch_lookup[selected_batch]
            known_students = db.get_students_by_batch(batch_id)
        with col_c2:
            selected_date = st.date_input("Lecture Date", value=date.today())
            if known_students:
                st.caption(f"🟢 **{len(known_students)}** registered student(s) ready for verification.")
            else:
                st.caption("⚠️ No students registered in this batch.")

    if not known_students:
        st.info(f"Please register students in **{selected_batch}** before taking attendance.")
        st.stop()

    st.space("small")

    with st.container(border=True):
        st.subheader(":material/photo_camera: Class Photo & Matching Configuration", text_alignment="left")
        
        col_mode, col_strict = st.columns([1, 1], gap="medium")
        with col_mode:
            capture_mode = st.segmented_control(
                "Photo Input Method",
                ["Camera", "Upload File"],
                default="Camera",
                key="att_mode_seg",
            )
            photo_file = None
            if capture_mode == "Camera":
                photo_file = st.camera_input("Capture group or individual photo", key="att_cam")
            else:
                photo_file = st.file_uploader("Upload class snapshot", type=["jpg", "jpeg", "png"], key="att_up")

        with col_strict:
            st.markdown("**:material/tune: Matching Sensitivity Preset**")
            strict_preset = st.segmented_control(
                "Preset Strictness",
                ["Strict (0.45)", "Balanced (0.50)", "Lenient (0.55)"],
                default="Balanced (0.50)",
            )
            
            preset_map = {
                "Strict (0.45)": 0.45,
                "Balanced (0.50)": 0.50,
                "Lenient (0.55)": 0.55,
            }
            tolerance = preset_map[strict_preset]
            
            with st.expander("Custom Strictness Fine-Tuning"):
                tolerance = st.slider(
                    "Match strictness threshold (lower = stricter)",
                    min_value=0.30,
                    max_value=0.70,
                    value=tolerance,
                    step=0.01,
                    help="Lower threshold reduces false positives, higher threshold increases sensitivity.",
                )

    if photo_file is not None:
        st.space("small")
        if st.button("Process Attendance", icon=":material/auto_awesome:", type="primary", width="stretch"):
            with st.spinner("Detecting facial features and verifying student profiles..."):
                img_np = image_to_np(photo_file)
                faces = fu.get_all_faces(img_np)

                if not faces:
                    st.error("No faces detected in the provided photo. Ensure good lighting and front-facing visibility.", icon=":material/error:")
                else:
                    present_ids = set()
                    for location, encoding in faces:
                        match = fu.match_face(encoding, known_students, tolerance=tolerance)
                        if match:
                            present_ids.add(match["id"])

                    date_str = selected_date.isoformat()
                    for s in known_students:
                        status = "Present" if s["id"] in present_ids else "Absent"
                        db.mark_attendance(s["id"], batch_id, date_str, status)

                    present_count = len(present_ids)
                    absent_count = len(known_students) - present_count
                    rate_pct = (present_count / len(known_students)) * 100

                    st.toast(f"Attendance recorded for {date_str}!", icon="✅")

                    # Results Dashboard
                    with st.container(border=True):
                        st.subheader(f":material/fact_check: Attendance Results ({date_str})", text_alignment="left")
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Attendance Rate", f"{rate_pct:.1f}%")
                        with m2:
                            st.metric("Present Students", present_count)
                        with m3:
                            st.metric("Absent Students", absent_count)

                        st.space("small")

                        # Visualization Chart & Table
                        col_tbl, col_chart = st.columns([3, 2])
                        
                        result_rows = [
                            {
                                "Student Name": s["name"],
                                "Roll Number": s["roll_no"],
                                "Status": "Present" if s["id"] in present_ids else "Absent",
                            }
                            for s in known_students
                        ]
                        df_res = pd.DataFrame(result_rows)

                        with col_tbl:
                            st.dataframe(
                                df_res,
                                column_config={
                                    "Student Name": st.column_config.TextColumn("Student Name", width="large"),
                                    "Roll Number": st.column_config.TextColumn("Roll No / ID", width="medium"),
                                    "Status": st.column_config.TextColumn("Status", width="medium"),
                                },
                                hide_index=True,
                                width="stretch",
                            )

                        with col_chart:
                            st.markdown("**:material/pie_chart: Class Distribution**")
                            chart_df = pd.DataFrame({
                                "Status": ["Present", "Absent"],
                                "Count": [present_count, absent_count]
                            }).set_index("Status")
                            st.bar_chart(chart_df, color=["#10B981" if present_count > 0 else "#EF4444"])


# ---------------- Page: View Reports ----------------

elif page == "Attendance Reports":
    st.header("Attendance Analytics & Reports", text_alignment="left")

    st.caption("Inspect historical attendance trends, student metrics, and download CSV reports.")

    if not batch_names:
        st.warning("Please create a batch first under **Dashboard & Batches**.", icon=":material/warning:")
        st.stop()

    st.space("small")

    report_type = st.segmented_control(
        "Report Mode",
        ["Date-wise Report", "Student-wise Analysis", "Batch Summary"],
        default="Date-wise Report",
    )

    st.space("small")

    selected_batch = st.selectbox("Select Batch", batch_names, key="rep_batch_sel")
    batch_id = batch_lookup[selected_batch]

    if report_type == "Date-wise Report":
        col_d, col_empty = st.columns([1, 2])
        with col_d:
            selected_date = st.date_input("Select Date", value=date.today(), key="rep_date_in")

        rows = db.get_attendance_by_date(batch_id, selected_date.isoformat())
        
        with st.container(border=True):
            st.subheader(f":material/calendar_today: Attendance for {selected_batch} ({selected_date})", text_alignment="left")
            if rows:
                df = pd.DataFrame(rows, columns=["Name", "Roll No", "Status", "Time Marked"])
                
                # Metrics header
                p_cnt = (df["Status"] == "Present").sum()
                total_cnt = len(df)
                p_rate = (p_cnt / total_cnt * 100) if total_cnt > 0 else 0
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Attendance Rate", f"{p_rate:.1f}%")
                with m2:
                    st.metric("Present", p_cnt)
                with m3:
                    st.metric("Absent", total_cnt - p_cnt)

                st.space("small")

                st.dataframe(
                    df,
                    column_config={
                        "Name": st.column_config.TextColumn("Student Name", width="large"),
                        "Roll No": st.column_config.TextColumn("Roll No", width="medium"),
                        "Status": st.column_config.TextColumn("Status", width="medium"),
                        "Time Marked": st.column_config.TextColumn("Time Logged", width="medium"),
                    },
                    hide_index=True,
                    width="stretch",
                )

                st.space("small")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Report (CSV)",
                    data=csv,
                    file_name=f"attendance_{selected_batch}_{selected_date}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    type="primary",
                )
            else:
                st.info(f"No attendance recorded for **{selected_batch}** on **{selected_date}**.")

    elif report_type == "Student-wise Analysis":
        students = db.get_students_by_batch(batch_id)
        if not students:
            st.info(f"No registered students found in **{selected_batch}**.")
        else:
            names_map = {f"{s['name']} ({s['roll_no']})": s["id"] for s in students}
            col_s, _ = st.columns([1, 2])
            with col_s:
                selected_student_label = st.selectbox("Select Student", list(names_map.keys()))
            
            s_id = names_map[selected_student_label]
            rows = db.get_attendance_by_student(s_id)
            
            with st.container(border=True):
                st.subheader(f":material/person: Profile History: {selected_student_label}", text_alignment="left")
                if rows:
                    df = pd.DataFrame(rows, columns=["Date", "Status", "Time Marked"])
                    present_cnt = (df["Status"] == "Present").sum()
                    total_cnt = len(df)
                    pct = (present_cnt / total_cnt * 100) if total_cnt > 0 else 0

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Overall Attendance", f"{pct:.1f}%")
                    with col_m2:
                        st.metric("Lectures Attended", f"{present_cnt} / {total_cnt}")
                    with col_m3:
                        st.metric("Absences", total_cnt - present_cnt)

                    st.space("small")

                    st.dataframe(
                        df,
                        column_config={
                            "Date": st.column_config.DateColumn("Date", width="medium"),
                            "Status": st.column_config.TextColumn("Status", width="medium"),
                            "Time Marked": st.column_config.TextColumn("Time Logged", width="medium"),
                        },
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info("No attendance history found for this student.")

    else:  # Batch Summary
        rows = db.get_attendance_summary(batch_id)
        with st.container(border=True):
            st.subheader(f":material/analytics: Comprehensive Batch Summary: {selected_batch}", text_alignment="left")
            if rows:
                df = pd.DataFrame(rows, columns=["Name", "Roll No", "Present", "Total Lectures"])
                df["Attendance %"] = (df["Present"] / df["Total Lectures"].replace(0, 1) * 100).round(1)

                avg_batch_att = df["Attendance %"].mean()
                top_student = df.sort_values(by="Attendance %", ascending=False).iloc[0]["Name"]

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Batch Average Attendance", f"{avg_batch_att:.1f}%")
                with m2:
                    st.metric("Highest Attending Student", top_student)

                st.space("small")

                st.dataframe(
                    df,
                    column_config={
                        "Name": st.column_config.TextColumn("Student Name", width="large"),
                        "Roll No": st.column_config.TextColumn("Roll No / ID", width="medium"),
                        "Present": st.column_config.NumberColumn("Lectures Present", format="%d 🟢"),
                        "Total Lectures": st.column_config.NumberColumn("Total Sessions", format="%d 📚"),
                        "Attendance %": st.column_config.ProgressColumn(
                            "Attendance Rate",
                            format="%.1f%%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    width="stretch",
                )

                st.space("small")
                st.markdown("**:material/bar_chart: Attendance Rate per Student**")
                chart_data = df.set_index("Name")[["Attendance %"]]
                st.bar_chart(chart_data)

                st.space("small")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Summary CSV",
                    data=csv,
                    file_name=f"summary_{selected_batch}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    type="primary",
                )
            else:
                st.info(f"No student data available for **{selected_batch}**.")
