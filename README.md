# Least Laxity First (LLF) Real-Time Scheduler Simulator

This project is an interactive, visual simulator for the **Least Laxity First (LLF)** / **Least Slack Time First (LST)** real-time scheduling algorithm. It is designed to model and schedule any arbitrary set of **aperiodic tasks** and visualize the scheduling process dynamically.

The application supports a complete bilingual interface (English and **فارسی/Persian**) with full **Right-to-Left (RTL)** layout shifts in Persian mode. It is built using **Python**, **Streamlit** for the web dashboard, and **Plotly** for interactive graphs.

---

## ⚙️ Theory & Mathematics

**Laxity** (or **Slack Time**) represents the amount of spare time a task has before it *must* execute to meet its deadline. For any active task $T_i$ at time step $t$, the laxity $L_i(t)$ is calculated as:

$$L_i(t) = D_i - t - C_i(t)$$

Where:
* $D_i$ is the absolute deadline of the task.
* $t$ is the current simulation time.
* $C_i(t)$ is the remaining execution/computation time required by the task at time $t$.

### Scheduling Decisions:
* The task with the **least laxity** is assigned the highest priority and scheduled to execute.
* In **preemptive** mode, laxity is evaluated at every integer time step. If a waiting task's laxity becomes smaller than the currently executing task's laxity, a preemption occurs.
* In **non-preemptive** mode, once a task starts running, it occupies the CPU until completion.
* **Tie-Breaker Policies**: In case multiple tasks share the same minimum laxity, the tie is broken using:
  * **Keep Currently Running** (prevents context switches and thrashing).
  * **Earlier Deadline**.
  * **Lower Task ID**.
* **Deadline Miss Policies**:
  * **Abort Immediately**: Aborts a task the moment it is guaranteed to miss its deadline ($L_i(t) < 0$).
  * **Run to Completion**: Allows tasks to finish even if they miss their deadlines (soft real-time).

---

## 🚀 How to Run the Project

You can run the project either locally (in a virtual environment) or using Docker.

### Method 1: Local Run (Python Virtual Environment)

1. Open your terminal in the project directory:
   ```bash
   cd /home/kshyst/Desktop/University/ES_HW6
   ```

2. Make the runner script executable (already configured):
   ```bash
   chmod +x run.sh
   ```

3. Start the application:
   ```bash
   ./run.sh
   ```
   This automatically activates the local virtual environment (`venv/`) and launches the Streamlit app. It will open in your default browser at:
   **http://localhost:8501**

---

### Method 2: Docker & Docker Compose (Recommended)

The project is fully containerized with a lightweight Debian-slim Python image and runs as a non-root system user for security.

1. **Build and start the container**:
   ```bash
   docker compose up -d --build
   ```

2. **Access the application**:
   Open **[http://localhost:8501](http://localhost:8501)** in your web browser.

3. **Check runtime logs**:
   ```bash
   docker compose logs -f
   ```

4. **Stop the container**:
   ```bash
   docker compose down
   ```

*(Note: If you run using Docker, make sure you don't have the local Streamlit process running on port 8501 to prevent port allocation conflicts).*

---

## 🌟 Key Application Features

1. **Bilingual RTL Layout**: toggling the language in the sidebar to **"فارسی"** dynamically mirrors the layout, moving the sidebar to the right and shifting text direction to RTL.
2. **Interactive Task Editor & 5 Scenarios**: Select **"Custom Manual Entry"** to dynamically add, edit, or delete tasks in a spreadsheet-like editor. It features a dropdown to load **5 pre-made examples** covering all scheduling behaviors:
   * **Classic Preemption**: Triggers multiple preemptions and tie-breakers.
   * **Thrashing Scenario**: Illustrates why the "Keep Currently Running" tie-breaker is essential to prevent infinite step-by-step preemption.
   * **Preemptive vs Non-Preemptive**: Compares scheduling feasibility between the two models.
   * **Tight Deadlines**: Visualizes the "Abort Immediately" policy vs. soft real-time completion.
   * **Sparse Arrivals**: Demonstrates CPU idle periods and active transitions.
3. **Random Task Generator**: Slide ranges for arrival time, computation time, and slack to test the scheduler on synthetic sets of tasks.
4. **Dynamic Gantt Chart**: an interactive timeline showing CPU execution states. It automatically scales vertically to list all tasks in the task set, showing empty tracks for tasks that were aborted before execution.
5. **Laxity-over-Time Plot**: charts $L_i(t)$ vs time, illustrating how slack values change and visualizing the $L = 0$ critical boundary line.
6. **Detailed Metrics**: counts CPU utilization, waiting times, context switches, preemptions, and missed deadlines.
7. **Step-by-Step Log**: an interactive log detailing calculations and scheduler choices at every single integer time unit.
