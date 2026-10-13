# Face Recognition Attendance System (Streamlit)

A batch-wise (Data Science / Web Development / C/C++ / custom), date-wise and
name-wise attendance system powered by face recognition.

## How it works

1. **Manage Batches** – create/view class batches.
2. **Register Student** – capture one clear photo per student; the app stores
   their face "encoding" (a 128-number fingerprint of their face), not the
   raw image match, so recognition is fast and doesn't need internet access.
3. **Take Attendance** – for a given batch and date, capture or upload a
   single class/group photo. The app detects every face in it, matches each
   against that batch's registered students, and marks Present/Absent for
   everyone automatically.
4. **View Reports** – date-wise register, per-student history with
   attendance %, and a full batch summary — all exportable as CSV.

Data is stored locally in a SQLite file (`attendance.db`) created next to
`app.py` the first time you run the app. No cloud service is used.

## Setup

**1. Install Python 3.9–3.11** (dlib, a dependency of `face_recognition`,
does not yet have prebuilt wheels for the very latest Python versions on
every OS).

**2. Install system dependency `cmake`** (required to build `dlib`):

- Windows: `pip install cmake` usually suffices, or install "CMake" from
  cmake.org and add it to PATH.
- macOS: `brew install cmake`
- Linux (Debian/Ubuntu): `sudo apt-get install cmake build-essential`

**3. Install Python packages:**

```bash
pip install -r requirements.txt
```

The default deployment uses the OpenCV vision engine. The optional
`face_recognition` package is not included in the deployment requirements
because its `dlib` dependency is not reliably available on Streamlit Cloud.
The app automatically uses OpenCV when `face_recognition` is unavailable.

> If `pip install face_recognition` fails while building `dlib`, install
> dlib first with conda (`conda install -c conda-forge dlib`) then re-run
> `pip install face_recognition`.

**4. Run the app:**

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Tips for good recognition accuracy

- Register students with even, front-facing lighting and no sunglasses/masks.
- For attendance, one wide, well-lit group photo works better than a blurry
  one — encourage students to face the camera.
- If two people keep getting swapped or matches are missed, adjust the
  **"Match strictness"** slider on the Take Attendance page (lower = fewer
  false positives, but may miss genuine matches; higher = more lenient).
- Recognition happens locally on your machine — no images are sent anywhere.

## File structure

```
face_attendance_system/
├── app.py           # Streamlit UI and page logic
├── database.py       # SQLite schema and queries (batches/students/attendance)
├── face_utils.py      # face_recognition wrappers (encode, detect, match)
├── requirements.txt
└── README.md
```

## Extending it further

- Add student photo removal / re-registration.
- Add an "Edit attendance" page to manually override a status.
- Swap SQLite for PostgreSQL if multiple people need to use it at once.
- Add authentication (e.g. `streamlit-authenticator`) so only faculty can
  mark attendance.
#
