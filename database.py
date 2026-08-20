"""
database.py
Handles all SQLite operations for the Face Recognition Attendance System.
Tables:
  - batches: id, name
  - students: id, name, roll_no, batch_id, encoding (pickled numpy array), photo BLOB
  - attendance: id, student_id, batch_id, date, status, time_marked
"""

import sqlite3
import pickle
from datetime import datetime

DB_PATH = "attendance.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_no TEXT NOT NULL,
            batch_id INTEGER NOT NULL,
            encoding BLOB NOT NULL,
            photo BLOB,
            UNIQUE(roll_no, batch_id),
            FOREIGN KEY (batch_id) REFERENCES batches (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            batch_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('Present', 'Absent')),
            time_marked TEXT,
            UNIQUE(student_id, date),
            FOREIGN KEY (student_id) REFERENCES students (id),
            FOREIGN KEY (batch_id) REFERENCES batches (id)
        )
    """)

    conn.commit()

    # Seed default batches if the table is empty
    cur.execute("SELECT COUNT(*) FROM batches")
    if cur.fetchone()[0] == 0:
        for b in ["Data Science", "Web Development", "Web Design"]:
            cur.execute("INSERT INTO batches (name) VALUES (?)", (b,))
        conn.commit()

    conn.close()


# ---------------- Batches ----------------

def get_batches():
    conn = get_connection()
    rows = conn.execute("SELECT id, name FROM batches ORDER BY name").fetchall()
    conn.close()
    return rows


def add_batch(name):
    conn = get_connection()
    try:
        conn.execute("INSERT INTO batches (name) VALUES (?)", (name,))
        conn.commit()
        return True, "Batch added."
    except sqlite3.IntegrityError:
        return False, "Batch already exists."
    finally:
        conn.close()


# ---------------- Students ----------------

def add_student(name, roll_no, batch_id, encoding, photo_bytes):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO students (name, roll_no, batch_id, encoding, photo) VALUES (?, ?, ?, ?, ?)",
            (name, roll_no, batch_id, pickle.dumps(encoding), photo_bytes),
        )
        conn.commit()
        return True, "Student registered."
    except sqlite3.IntegrityError:
        return False, "This roll number already exists in this batch."
    finally:
        conn.close()


def get_students_by_batch(batch_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, roll_no, encoding FROM students WHERE batch_id = ? ORDER BY name",
        (batch_id,),
    ).fetchall()
    conn.close()
    students = []
    for r in rows:
        students.append({
            "id": r[0],
            "name": r[1],
            "roll_no": r[2],
            "encoding": pickle.loads(r[3]),
        })
    return students


def get_all_students():
    conn = get_connection()
    rows = conn.execute("""
        SELECT students.id, students.name, students.roll_no, batches.name
        FROM students JOIN batches ON students.batch_id = batches.id
        ORDER BY batches.name, students.name
    """).fetchall()
    conn.close()
    return rows

def delete_student(student_id):
    conn = get_connection()
    conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()


# ---------------- Attendance ----------------

def mark_attendance(student_id, batch_id, date, status):
    conn = get_connection()
    conn.execute(
        """INSERT INTO attendance (student_id, batch_id, date, status, time_marked)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(student_id, date) DO UPDATE SET status=excluded.status, time_marked=excluded.time_marked""",
        (student_id, batch_id, date, status, datetime.now().strftime("%H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_attendance_by_date(batch_id, date):
    conn = get_connection()
    rows = conn.execute("""
        SELECT students.name, students.roll_no, attendance.status, attendance.time_marked
        FROM attendance JOIN students ON attendance.student_id = students.id
        WHERE attendance.batch_id = ? AND attendance.date = ?
        ORDER BY students.name
    """, (batch_id, date)).fetchall()
    conn.close()
    return rows


def get_attendance_by_student(student_id):
    conn = get_connection()
    rows = conn.execute("""
        SELECT date, status, time_marked FROM attendance
        WHERE student_id = ? ORDER BY date DESC
    """, (student_id,)).fetchall()
    conn.close()
    return rows


def get_attendance_summary(batch_id):
    """Returns per-student present/total counts for a batch, for a quick % view."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT students.name, students.roll_no,
               SUM(CASE WHEN attendance.status = 'Present' THEN 1 ELSE 0 END) AS present_count,
               COUNT(attendance.id) AS total_marked
        FROM students LEFT JOIN attendance ON students.id = attendance.student_id
        WHERE students.batch_id = ?
        GROUP BY students.id
        ORDER BY students.name
    """, (batch_id,)).fetchall()
    conn.close()
    return rows


def get_stats_summary():
    """Returns total batches, total students, and today's attendance count."""
    conn = get_connection()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    total_batches = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    today_present = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Present'", (today_str,)
    ).fetchone()[0]
    today_total = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ?", (today_str,)
    ).fetchone()[0]
    
    conn.close()
    return {
        "batches": total_batches,
        "students": total_students,
        "today_present": today_present,
        "today_total": today_total,
    }


def delete_batch(batch_id):
    conn = get_connection()
    # Delete student attendance first, then students, then batch
    conn.execute("DELETE FROM attendance WHERE batch_id = ?", (batch_id,))
    conn.execute("DELETE FROM students WHERE batch_id = ?", (batch_id,))
    conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
    conn.commit()
    conn.close()
    return True, "Batch deleted successfully."

