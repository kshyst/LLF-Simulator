import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random

# Set Page Config
st.set_page_config(
    page_title="Least Laxity First (LLF) Scheduler Visualizer",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Rules for a Premium UI Look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    h2 {
        color: #2563EB;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 5px;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 4px;
    }
    .metric-lbl {
        font-size: 14px;
        font-weight: 500;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Core LLF Simulation Engine -----------------

def run_llf_simulation(tasks, preemptive, tie_breaker, deadline_policy):
    """
    Simulates Least Laxity First (LLF) scheduling.
    
    tasks: list of dicts: {'id': str, 'arrival': int, 'computation': int, 'deadline': int}
    preemptive: bool (True for preemptive, False for non-preemptive)
    tie_breaker: str ('Keep Currently Running', 'Earlier Deadline', 'Lower Task ID')
    deadline_policy: str ('Abort Immediately', 'Run to completion')
    """
    # Sort tasks initially by arrival time for standard reference
    tasks_sorted = sorted(tasks, key=lambda x: x['arrival'])
    
    # Track task state dynamically
    rem_comp = {t['id']: t['computation'] for t in tasks_sorted}
    arrival = {t['id']: t['arrival'] for t in tasks_sorted}
    deadline = {t['id']: t['deadline'] for t in tasks_sorted}
    orig_computation = {t['id']: t['computation'] for t in tasks_sorted}
    
    status = {t['id']: 'Pending' for t in tasks_sorted}
    start_times = {t['id']: [] for t in tasks_sorted}
    completion_time = {t['id']: None for t in tasks_sorted}
    abort_time = {t['id']: None for t in tasks_sorted}
    waiting_time = {t['id']: 0 for t in tasks_sorted}
    
    gantt_intervals = []
    laxity_history = []
    log = []
    
    current_time = 0
    prev_running_task = None
    context_switches = 0
    preemptions = 0
    
    # Calculate a safe upper bound for simulation length
    # Max arrival time + sum of computation times + additional buffer
    if not tasks_sorted:
        return [], [], pd.DataFrame(), [], 0, 0, 0, 0, 0
    
    max_sim_time = max(arrival.values()) + sum(orig_computation.values()) + 100
    
    while current_time < max_sim_time:
        # Check if all tasks are finished or aborted
        all_done = True
        for tid in rem_comp:
            if status[tid] not in ['Completed', 'Aborted']:
                all_done = False
                break
        if all_done:
            break
            
        # 1. Handle newly arrived tasks
        for tid in rem_comp:
            if status[tid] == 'Pending' and current_time >= arrival[tid]:
                status[tid] = 'Ready'
                log.append(f"<b>Time {current_time}:</b> Task <span style='color:#2563EB; font-weight:bold'>{tid}</span> arrived (Computation needed: {rem_comp[tid]}, Deadline: {deadline[tid]})")
                
        # 2. Check for deadline misses
        active_ready_tasks = []
        for tid in rem_comp:
            if status[tid] in ['Ready', 'Running']:
                # Laxity (Slack) = Deadline - Current Time - Remaining Computation
                lax = deadline[tid] - current_time - rem_comp[tid]
                
                # Check if it has missed its deadline or is guaranteed to miss it
                if deadline_policy == 'Abort Immediately' and lax < 0:
                    status[tid] = 'Aborted'
                    abort_time[tid] = current_time
                    rem_comp[tid] = 0
                    log.append(f"<b>Time {current_time}:</b> Task <span style='color:#EF4444; font-weight:bold'>{tid}</span> <b>ABORTED</b> (Guaranteed deadline miss. Laxity: {lax})")
                elif current_time >= deadline[tid]:
                    if deadline_policy == 'Abort Immediately':
                        status[tid] = 'Aborted'
                        abort_time[tid] = current_time
                        rem_comp[tid] = 0
                        log.append(f"<b>Time {current_time}:</b> Task <span style='color:#EF4444; font-weight:bold'>{tid}</span> <b>ABORTED</b> (Deadline {deadline[tid]} reached, task incomplete)")
                    else:
                        # For soft real-time, keep running it
                        active_ready_tasks.append(tid)
                else:
                    active_ready_tasks.append(tid)
            elif status[tid] == 'Pending' and current_time >= deadline[tid]:
                # Arrived past its deadline (edge case)
                if deadline_policy == 'Abort Immediately':
                    status[tid] = 'Aborted'
                    abort_time[tid] = current_time
                    rem_comp[tid] = 0
                    log.append(f"<b>Time {current_time}:</b> Task <span style='color:#EF4444; font-weight:bold'>{tid}</span> aborted before release due to deadline miss")
        
        # 3. Calculate laxity for all active ready tasks and record to history
        laxities = {}
        for tid in active_ready_tasks:
            lax = deadline[tid] - current_time - rem_comp[tid]
            laxities[tid] = lax
            laxity_history.append({'Time': current_time, 'Task': tid, 'Laxity': lax})
            
        # 4. Select task to run
        selected_task = None
        
        if not active_ready_tasks:
            selected_task = None
            if any(status[t] == 'Pending' for t in status):
                log.append(f"<b>Time {current_time}:</b> CPU is Idle (Waiting for tasks to arrive)")
            else:
                log.append(f"<b>Time {current_time}:</b> CPU is Idle")
        else:
            # If non-preemptive and a task is already running and active, continue executing it
            if not preemptive and prev_running_task in active_ready_tasks:
                selected_task = prev_running_task
            else:
                # Find minimum laxity value
                min_lax = min(laxities.values())
                candidates = [tid for tid, lax in laxities.items() if lax == min_lax]
                
                if len(candidates) == 1:
                    selected_task = candidates[0]
                else:
                    # Apply Tie-breaking rules
                    tie_desc = f"<b>Time {current_time}:</b> Laxity tie ({min_lax}) between {candidates}."
                    if tie_breaker == 'Keep Currently Running' and prev_running_task in candidates:
                        selected_task = prev_running_task
                        tie_desc += f" Selected <span style='color:#10B981; font-weight:bold'>{selected_task}</span> (Currently Running to avoid context switch)"
                    elif tie_breaker == 'Earlier Deadline':
                        # Sort candidates by deadline, then ID as secondary fallback
                        candidates_sorted = sorted(candidates, key=lambda x: (deadline[x], x))
                        selected_task = candidates_sorted[0]
                        tie_desc += f" Selected <span style='color:#10B981; font-weight:bold'>{selected_task}</span> (Earlier Deadline: {deadline[selected_task]})"
                    else: # 'Lower Task ID'
                        candidates_sorted = sorted(candidates)
                        selected_task = candidates_sorted[0]
                        tie_desc += f" Selected <span style='color:#10B981; font-weight:bold'>{selected_task}</span> (Lower Task ID)"
                    log.append(tie_desc)
                    
        # 5. Execute the selected task
        if selected_task:
            status[selected_task] = 'Running'
            if not start_times[selected_task] or (gantt_intervals and gantt_intervals[-1]['Task'] != selected_task):
                start_times[selected_task].append(current_time)
            
            # Identify preemption and context switch
            if prev_running_task != selected_task:
                if prev_running_task is not None:
                    context_switches += 1
                    # Was the previous task preempted (still has execution time and not finished/aborted)?
                    if rem_comp[prev_running_task] > 0 and status[prev_running_task] in ['Running', 'Ready']:
                        preemptions += 1
                        prev_lax = deadline[prev_running_task] - current_time - rem_comp[prev_running_task]
                        log.append(f"<b>Time {current_time}:</b> Task <span style='color:#10B981; font-weight:bold'>{selected_task}</span> (Laxity: {laxities[selected_task]}) <b>PREEMPTS</b> Task <span style='color:#F59E0B; font-weight:bold'>{prev_running_task}</span> (Laxity: {prev_lax})")
                        status[prev_running_task] = 'Ready'
                    else:
                        log.append(f"<b>Time {current_time}:</b> Task <span style='color:#10B981; font-weight:bold'>{selected_task}</span> starts executing (Task <span style='color:#F59E0B; font-weight:bold'>{prev_running_task}</span> finished/aborted)")
                else:
                    log.append(f"<b>Time {current_time}:</b> Task <span style='color:#10B981; font-weight:bold'>{selected_task}</span> starts executing after CPU idle period")
            
            # Decrement execution time
            rem_comp[selected_task] -= 1
            
            # Append/merge Gantt interval
            if gantt_intervals and gantt_intervals[-1]['Task'] == selected_task and gantt_intervals[-1]['Finish'] == current_time:
                gantt_intervals[-1]['Finish'] = current_time + 1
            else:
                gantt_intervals.append({'Task': selected_task, 'Start': current_time, 'Finish': current_time + 1})
                
            # If completed
            if rem_comp[selected_task] == 0:
                status[selected_task] = 'Completed'
                completion_time[selected_task] = current_time + 1
                log.append(f"<b>Time {current_time+1}:</b> Task <span style='color:#10B981; font-weight:bold'>{selected_task}</span> <b>COMPLETED</b>")
        else:
            # Idle Gantt block
            if gantt_intervals and gantt_intervals[-1]['Task'] == 'Idle' and gantt_intervals[-1]['Finish'] == current_time:
                gantt_intervals[-1]['Finish'] = current_time + 1
            else:
                gantt_intervals.append({'Task': 'Idle', 'Start': current_time, 'Finish': current_time + 1})
                
        # 6. Increment waiting time for ready tasks that did not run
        for tid in active_ready_tasks:
            if tid != selected_task:
                waiting_time[tid] += 1
                
        prev_running_task = selected_task if (selected_task and rem_comp[selected_task] > 0) else None
        current_time += 1

    # Format Results Table
    results = []
    for t in tasks_sorted:
        tid = t['id']
        arr = arrival[tid]
        comp = orig_computation[tid]
        dl = deadline[tid]
        
        c_time = completion_time[tid]
        a_time = abort_time[tid]
        w_time = waiting_time[tid]
        
        if c_time is not None:
            lateness = c_time - dl
            tardiness = max(0, lateness)
            final_status = "Completed on Time" if lateness <= 0 else f"Completed Late (+{lateness})"
        elif a_time is not None:
            lateness = None
            tardiness = None
            final_status = f"Aborted at {a_time} (Missed)"
        else:
            lateness = None
            tardiness = None
            final_status = "Unfinished"
            
        first_start = start_times[tid][0] if start_times[tid] else None
        
        results.append({
            'Task ID': tid,
            'Arrival Time': arr,
            'Computation Time': comp,
            'Deadline': dl,
            'First Start Time': first_start,
            'Completion Time': c_time,
            'Waiting Time': w_time,
            'Lateness': lateness,
            'Tardiness': tardiness,
            'Status': final_status
        })
        
    df_results = pd.DataFrame(results)
    
    # Overall Performance Metrics
    total_time = current_time
    idle_time = sum(interval['Finish'] - interval['Start'] for interval in gantt_intervals if interval['Task'] == 'Idle')
    busy_time = total_time - idle_time
    cpu_utilization = (busy_time / total_time) * 100 if total_time > 0 else 0
    
    missed_deadlines = sum(1 for r in results if "Late" in str(r['Status']) or "Aborted" in str(r['Status']))
    avg_waiting = df_results['Waiting Time'].mean() if len(df_results) > 0 else 0
    
    return gantt_intervals, laxity_history, df_results, log, cpu_utilization, avg_waiting, context_switches, preemptions, missed_deadlines

# ----------------- Streamlit UI Layout -----------------

# Language Toggle (English / Persian)
lang = st.sidebar.radio("Language / زبان", ["English", "فارسی"], index=0)

# Apply RTL Layout conditionally for Persian (فارسی)
if lang == "فارسی":
    st.markdown("""
    <style>
        /* Force Right-to-Left Layout and swap the sidebar position */
        [data-testid="stAppViewContainer"] {
            flex-direction: row-reverse !important;
        }
        [data-testid="stSidebar"] {
            direction: rtl !important;
            text-align: right !important;
        }
        .main .block-container {
            direction: rtl !important;
            text-align: right !important;
        }
        /* Swap the collapsed sidebar controller position to top-right */
        [data-testid="stSidebarCollapsedControl"] {
            left: auto !important;
            right: 0 !important;
            padding: 8px !important;
        }
        /* Ensure all core typographic and alert structures align RTL */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stAlert, li {
            text-align: right !important;
            direction: rtl !important;
        }
        /* Fix input labels and placeholders */
        .stTextInput input, .stNumberInput input, .stSelectbox div {
            text-align: right !important;
            direction: rtl !important;
        }
        /* Fix table alignments */
        .stDataFrame, .stTable {
            direction: rtl !important;
        }
        /* Align specific widget items like checkboxes and radios */
        [data-testid="stWidgetLabel"] p {
            text-align: right !important;
        }
        /* Swap border highlight side for metric cards in RTL */
        .metric-card {
            border-left: 1px solid #E2E8F0 !important;
            border-right: 5px solid #2563EB !important;
        }
    </style>
    """, unsafe_allow_html=True)

if lang == "English":
    title = "⏰ Least Laxity First (LLF) Scheduler Visualizer"
    intro_text = """
    This tool simulates and visualizes the **Least Laxity First (LLF)** (also known as **Least Slack Time - LST**) 
    real-time scheduling algorithm for a set of arbitrary aperiodic tasks.
    
    **Laxity (Slack Time)** represents the amount of spare time a task has before it *must* execute to meet its deadline. 
    It is computed at any time $t$ for an active task $T_i$ as:
    $$L_i(t) = D_i - t - C_i(t)$$
    where:
    * $D_i$ is the absolute deadline of the task.
    * $t$ is the current simulation time.
    * $C_i(t)$ is the remaining execution/computation time needed.
    """
    sidebar_title = "Simulation Config"
    preemption_lbl = "Preemption Mode"
    preemptive_choice = ["Preemptive", "Non-Preemptive"]
    tie_lbl = "Tie-Breaking Policy"
    tie_choices = ["Keep Currently Running", "Earlier Deadline", "Lower Task ID"]
    deadline_lbl = "Deadline Miss Policy"
    deadline_choices = ["Abort Immediately", "Run to completion"]
    input_lbl = "Task Input Method"
    input_choices = ["Custom Manual Entry", "Random Task Generator"]
    
    t_id = "Task ID"
    t_arr = "Arrival Time"
    t_comp = "Computation Time"
    t_dl = "Deadline"
    
    btn_generate = "Generate Random Tasks"
    btn_simulate = "Simulate Scheduling"
    results_lbl = "Simulation Results"
    
    tab_gantt = "Gantt Chart"
    tab_laxity = "Laxity Over Time"
    tab_data = "Task Data Table"
    tab_log = "Step-by-Step Log"
    tab_theory = "Scheduling Theory"
    
    metric_cpu = "CPU Utilization"
    metric_avg_wait = "Average Waiting Time"
    metric_switches = "Context Switches"
    metric_preempts = "Preemptions (Involuntary)"
    metric_missed = "Missed Deadlines"
else:
    title = "⏰ شبیه‌ساز و تصویرساز الگوریتم زمان‌بندی کمترین سستی (LLF)"
    intro_text = """
    این ابزار الگوریتم زمان‌بندی زمان‌واقعی **کمترین سستی (Least Laxity First - LLF)** یا **کمترین زمان استراحت (Least Slack Time - LST)** را برای مجموعه‌ای از وظایف غیردوره‌ای (aperiodic) دلخواه شبیه‌سازی و تصویرسازی می‌کند.
    
    **سستی (Laxity / Slack Time)** نشان‌دهنده مقدار زمان اضافی است که یک وظیفه دارد قبل از اینکه *باید* برای رسیدن به ضرب‌الاجل خود اجرا شود. سستی در هر لحظه $t$ برای وظیفه فعال $T_i$ به این صورت محاسبه می‌شود:
    $$L_i(t) = D_i - t - C_i(t)$$
    که در آن:
    * $D_i$ ضرب‌الاجل مطلق وظیفه است.
    * $t$ زمان فعلی شبیه‌سازی است.
    * $C_i(t)$ زمان محاسباتی/اجرایی باقی‌مانده مورد نیاز است.
    """
    sidebar_title = "پیکربندی شبیه‌سازی"
    preemption_lbl = "حالت پیش‌خوری (Preemption)"
    preemptive_choice = ["پیش‌خورانه (Preemptive)", "غیر پیش‌خورانه (Non-Preemptive)"]
    tie_lbl = "سیاست شکستن تساوی سستی"
    tie_choices = ["حفظ وظیفه در حال اجرای فعلی (کاهش تعویض‌قالب)", "ضرب‌الاجل نزدیک‌تر", "شناسه وظیفه کوچک‌تر"]
    deadline_lbl = "سیاست برخورد با عبور از ضرب‌الاجل"
    deadline_choices = ["سقط فوری وظیفه (Abort)", "ادامه اجرا تا تکمیل (Run to completion)"]
    input_lbl = "روش ورود اطلاعات وظایف"
    input_choices = ["ورود دستی وظایف دلخواه", "تولیدکننده وظایف تصادفی"]
    
    t_id = "شناسه وظیفه"
    t_arr = "زمان ورود (Arrival)"
    t_comp = "زمان محاسبات (Computation)"
    t_dl = "ضرب‌الاجل (Deadline)"
    
    btn_generate = "تولید وظایف تصادفی"
    btn_simulate = "شبیه‌سازی زمان‌بندی"
    results_lbl = "نتایج شبیه‌سازی"
    
    tab_gantt = "نمودار گانت (Gantt)"
    tab_laxity = "تغییرات سستی در طول زمان"
    tab_data = "جدول نتایج وظایف"
    tab_log = "گزارش گام‌به‌گام زمان‌بند"
    tab_theory = "تئوری علمی الگوریتم"
    
    metric_cpu = "بهره‌وری پردازنده (CPU Utilization)"
    metric_avg_wait = "میانگین زمان انتظار"
    metric_switches = "تعویض‌های قالب (Context Switches)"
    metric_preempts = "پیش‌خوری‌های ناخواسته (Preemptions)"
    metric_missed = "ضرب‌الاجل‌های از دست رفته"

st.title(title)
st.markdown(intro_text)

# ----------------- Sidebar Configuration -----------------

st.sidebar.header(sidebar_title)

# Preemption Setting
preempt_input = st.sidebar.selectbox(preemption_lbl, preemptive_choice)
preemptive = (preempt_input == preemptive_choice[0])

# Tie-Breaking Setting
tie_input = st.sidebar.selectbox(tie_lbl, tie_choices)
if tie_input == tie_choices[0]:
    tie_breaker = 'Keep Currently Running'
elif tie_input == tie_choices[1]:
    tie_breaker = 'Earlier Deadline'
else:
    tie_breaker = 'Lower Task ID'

# Deadline Miss Policy Setting
deadline_input = st.sidebar.selectbox(deadline_lbl, deadline_choices)
deadline_policy = 'Abort Immediately' if deadline_input == deadline_choices[0] else 'Run to completion'

# Input Method Setting
input_method = st.sidebar.radio(input_lbl, input_choices)

# ----------------- Task Input Handlers -----------------

tasks = []

# Default Classic Example Tasks
default_tasks = [
    {'id': 'T1', 'arrival': 0, 'computation': 4, 'deadline': 10},
    {'id': 'T2', 'arrival': 1, 'computation': 3, 'deadline': 6},
    {'id': 'T3', 'arrival': 2, 'computation': 2, 'deadline': 5},
    {'id': 'T4', 'arrival': 4, 'computation': 1, 'deadline': 8}
]

if input_method == input_choices[0]:  # Custom Manual Entry
    st.subheader("Enter Task Details" if lang == "English" else "ورود مشخصات وظایف")
    st.markdown(
        "Double-click cells to edit. Press `+` below the table to add new tasks." 
        if lang == "English" 
        else "روی سلول‌ها دوبار کلیک کنید تا ویرایش شوند. با دکمه `+` در زیر جدول می‌توانید وظایف جدید اضافه کنید."
    )
    
    # Initialize session state for manual tasks if not present to persist user edits across re-runs
    if "manual_tasks" not in st.session_state:
        st.session_state.manual_tasks = pd.DataFrame(default_tasks)
    
    edited_df = st.data_editor(
        st.session_state.manual_tasks,
        num_rows="dynamic",
        key="manual_tasks_editor",
        column_config={
            "id": st.column_config.TextColumn(t_id, default="T1", required=True),
            "arrival": st.column_config.NumberColumn(t_arr, default=0, min_value=0, step=1, required=True),
            "computation": st.column_config.NumberColumn(t_comp, default=1, min_value=1, step=1, required=True),
            "deadline": st.column_config.NumberColumn(t_dl, default=2, min_value=1, step=1, required=True),
        },
        use_container_width=True
    )
    
    # Persist the latest user changes (including new rows and edits) in session state
    st.session_state.manual_tasks = edited_df
    
    # Validate and build tasks list
    validation_ok = True
    for index, row in edited_df.iterrows():
        if pd.isna(row['id']) or str(row['id']).strip() == "":
            validation_ok = False
            st.error(f"Task at index {index} has an empty ID!" if lang == "English" else f"وظیفه در ردیف {index} فاقد شناسه است!")
        if row['computation'] < 1:
            validation_ok = False
            st.error(f"Task {row['id']} computation time must be >= 1!" if lang == "English" else f"زمان محاسبات وظیفه {row['id']} باید حداقل ۱ باشد!")
        if row['deadline'] < row['arrival'] + row['computation']:
            st.warning(
                f"Note: Task {row['id']} deadline is less than arrival + computation. This task is inherently infeasible!" 
                if lang == "English" 
                else f"توجه: ضرب‌الاجل وظیفه {row['id']} کمتر از زمان ورود به علاوه زمان اجراست. این وظیفه ذاتا غیرقابل انجام است!"
            )
            
    # Check for duplicate IDs
    if len(edited_df['id'].unique()) != len(edited_df):
        validation_ok = False
        st.error("Task IDs must be unique!" if lang == "English" else "شناسه‌های وظایف باید منحصربه‌فرد باشند!")
        
    if validation_ok and len(edited_df) > 0:
        tasks = edited_df.to_dict('records')
    else:
        st.info("Please resolve table errors to start simulation." if lang == "English" else "لطفاً خطاهای جدول را برطرف کنید تا شبیه‌سازی شروع شود.")

else:  # Random Generator
    st.subheader("Configure Generator" if lang == "English" else "پیکربندی تولیدکننده تصادفی")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        num_tasks = st.slider("Number of Tasks" if lang == "English" else "تعداد وظایف", 2, 8, 4)
    with c2:
        max_arr = st.slider("Max Arrival Time" if lang == "English" else "حداکثر زمان ورود", 0, 15, 6)
    with c3:
        comp_min, comp_max = st.slider("Computation Range" if lang == "English" else "محدوده زمان محاسبات", 1, 10, (2, 5))
    with c4:
        slack_min, slack_max = st.slider("Deadline Slack Range" if lang == "English" else "محدوده سستی ضرب‌الاجل", 0, 15, (2, 8))
        
    # Persistent random state using streamlit session state
    if "random_tasks" not in st.session_state or st.button(btn_generate):
        gen_tasks = []
        for i in range(num_tasks):
            arr = random.randint(0, max_arr)
            comp = random.randint(comp_min, comp_max)
            slack = random.randint(slack_min, slack_max)
            dl = arr + comp + slack
            gen_tasks.append({
                'id': f"T{i+1}",
                'arrival': arr,
                'computation': comp,
                'deadline': dl
            })
        st.session_state.random_tasks = gen_tasks
        
    tasks = st.session_state.random_tasks.copy()
    
    st.markdown("### Generated Tasks" if lang == "English" else "### وظایف تولید شده")
    df_show = pd.DataFrame(tasks).rename(columns={
        'id': t_id,
        'arrival': t_arr,
        'computation': t_comp,
        'deadline': t_dl
    })
    st.table(df_show)

# ----------------- Run Simulation -----------------

if len(tasks) > 0:
    gantt_intervals, laxity_history, df_results, log, cpu_util, avg_wait, switches, preempts, missed = run_llf_simulation(
        tasks, preemptive, tie_breaker, deadline_policy
    )
    
    st.header(results_lbl)
    
    # ----------------- Metrics Cards -----------------
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{cpu_util:.1f}%</div>
            <div class="metric-lbl">{metric_cpu}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{avg_wait:.2f}</div>
            <div class="metric-lbl">{metric_avg_wait}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{switches}</div>
            <div class="metric-lbl">{metric_switches}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{preempts}</div>
            <div class="metric-lbl">{metric_preempts}</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        color = "#EF4444" if missed > 0 else "#10B981"
        border_accent = f"border-right-color: {color} !important; border-left-color: #E2E8F0 !important; border-left-width: 1px !important; border-right-width: 5px !important;" if lang == "فارسی" else f"border-left-color: {color} !important;"
        st.markdown(f"""
        <div class="metric-card" style="{border_accent}">
            <div class="metric-val" style="color: {color}">{missed}</div>
            <div class="metric-lbl">{metric_missed}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("") # Spacer
    
    # ----------------- Visualizations & Logs (Tabs) -----------------
    
    t_gantt, t_laxity, t_data, t_log, t_theory = st.tabs([tab_gantt, tab_laxity, tab_data, tab_log, tab_theory])
    
    with t_gantt:
        st.subheader("Gantt Chart (Timeline of Task Execution)" if lang == "English" else "نمودار گانت (زمان‌بندی اجرای وظایف)")
        
        # Convert gantt intervals to a dataframe
        df_gantt = pd.DataFrame(gantt_intervals)
        df_gantt['Duration'] = df_gantt['Finish'] - df_gantt['Start']
        
        # Define color map for consistency
        all_unique_tasks = sorted(list(df_gantt['Task'].unique()))
        
        # Standard cool colors palette
        colors = px.colors.qualitative.Safe
        color_map = {}
        color_idx = 0
        for task_name in all_unique_tasks:
            if task_name == 'Idle':
                color_map['Idle'] = '#E2E8F0' # Soft light grey
            else:
                color_map[task_name] = colors[color_idx % len(colors)]
                color_idx += 1
                
        # Draw Horizontal Bar Chart using Plotly
        fig_gantt = px.bar(
            df_gantt,
            x="Duration",
            y="Task",
            base="Start",
            orientation="h",
            color="Task",
            color_discrete_map=color_map,
            category_orders={"Task": all_unique_tasks[::-1]}, # Idle at the bottom
            labels={"Task": "Task / وظیفه", "Start": "Time / زمان", "Duration": "Duration / مدت"},
            hover_data={"Start": True, "Finish": True, "Duration": True, "Task": False}
        )
        
        # Customize Grid lines and ticks
        max_time = df_gantt['Finish'].max()
        fig_gantt.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                range=[0, max_time + 0.5],
                title="Time / زمان"
            ),
            yaxis_title="Task / وظیفه",
            showlegend=True,
            legend_title_text="Tasks",
            height=300 + (len(all_unique_tasks) * 30),
            plot_bgcolor='white',
            margin=dict(l=50, r=50, t=20, b=50)
        )
        
        fig_gantt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9')
        fig_gantt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9')
        
        st.plotly_chart(fig_gantt, use_container_width=True)
        
        # Add visual deadline markers on the Gantt chart for educational clarity
        st.markdown(
            "*(Vertical grid lines represent 1-unit time steps. Visual markers representing deadlines can be cross-referenced with the data table.)*"
            if lang == "English"
            else "*(خطوط عمودی شبکه نشان‌دهنده گام‌های زمانی تک‌واحدی هستند. می‌توانید زمان‌های ضرب‌الاجل را با مراجعه به جدول زیر تطبیق دهید.)*"
        )
        
    with t_laxity:
        st.subheader("Dynamic Laxity (Slack Time) Over Time" if lang == "English" else "تغییرات پویای مقدار سستی در طول زمان")
        st.markdown(
            """
            This plot shows the remaining **Laxity (Slack)** of each ready task at each time step.
            
            * **Laxity remains constant** when a task is executing (the curve is horizontal).
            * **Laxity decreases by 1** every step a task is ready but waiting (the curve slopes downwards).
            * When a task's laxity drops to **0** (represented by the dashed red line), it has zero slack. It *must* run immediately, or it will miss its deadline.
            """
            if lang == "English"
            else """
            این نمودار میزان **سستی (Laxity)** باقی‌مانده از هر وظیفه آماده به کار را در هر گام زمانی نشان می‌دهد.
            
            * **سستی ثابت می‌ماند** وقتی که وظیفه در حال اجراست (خط افقی است).
            * **سستی به میزان ۱ واحد کاهش می‌یابد** به ازای هر گامی که وظیفه آماده است ولی منتظر اجرا می‌ماند (خط با شیب منفی به سمت پایین حرکت می‌کند).
            * هنگامی که سستی وظیفه به مرز **۰** (خط‌چین قرمز) برسد، وظیفه دیگر زمان استراحت ندارد و *باید* فوراً اجرا شود، در غیر این صورت از ضرب‌الاجل عبور خواهد کرد.
            """
        )
        
        if laxity_history:
            df_lax = pd.DataFrame(laxity_history)
            
            fig_lax = px.line(
                df_lax,
                x="Time",
                y="Laxity",
                color="Task",
                markers=True,
                color_discrete_map=color_map,
                labels={"Time": "Time / زمان", "Laxity": "Laxity (Slack) / سستی"},
                hover_data={"Time": True, "Laxity": True, "Task": True}
            )
            
            # Draw Critical Zero-Laxity boundary line
            max_t = df_lax['Time'].max()
            min_l = df_lax['Laxity'].min()
            max_l = df_lax['Laxity'].max()
            
            fig_lax.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=max_t + 1,
                y1=0,
                line=dict(color="#EF4444", width=2, dash="dash"),
                name="Zero Laxity Boundary"
            )
            
            # Annotate boundary
            fig_lax.add_annotation(
                x=max_t * 0.9,
                y=0.5 if max_l > 0 else -0.5,
                text="Zero Laxity / مرز سستی صفر",
                showarrow=False,
                font=dict(color="#EF4444", size=11, family="Inter, Arial")
            )
            
            fig_lax.update_layout(
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    range=[0, max_t + 1],
                    title="Time / زمان"
                ),
                yaxis=dict(
                    dtick=1,
                    title="Laxity / سستی"
                ),
                plot_bgcolor='white',
                height=450,
                margin=dict(l=50, r=50, t=25, b=50)
            )
            
            fig_lax.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9')
            fig_lax.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9')
            
            st.plotly_chart(fig_lax, use_container_width=True)
        else:
            st.info("No laxity history recorded. (Did tasks get executed without waiting?)" if lang == "English" else "تاریخچه سستی وجود ندارد. (آیا تمام وظایف بدون انتظار فوراً اجرا شدند؟)")
            
    with t_data:
        st.subheader("Detailed Scheduling Statistics per Task" if lang == "English" else "جزئیات آماری زمان‌بندی به تفکیک هر وظیفه")
        
        # Format the display names
        df_display = df_results.rename(columns={
            'Task ID': t_id,
            'Arrival Time': t_arr,
            'Computation Time': t_comp,
            'Deadline': t_dl,
            'First Start Time': "First Start Time / زمان اولین شروع",
            'Completion Time': "Completion Time / زمان پایان",
            'Waiting Time': "Waiting Time / زمان انتظار کل",
            'Lateness': "Lateness / تاخیر ضرب‌الاجل (با علامت)",
            'Tardiness': "Tardiness / مقدار دیرکرد مطلق",
            'Status': "Status / وضعیت"
        })
        
        st.dataframe(df_display.style.highlight_max(axis=0, color="#FFE4E6", subset=["Tardiness / مقدار دیرکرد مطلق"]), use_container_width=True)
        
    with t_log:
        st.subheader("Scheduling Trace Log" if lang == "English" else "شرح گزارش گام‌به‌گام تصمیمات زمان‌بند")
        st.markdown(
            "This log traces the scheduler state, laxity calculation, preemption, and completed tasks at each step."
            if lang == "English"
            else "این گزارش وضعیت زمان‌بند، محاسبات سستی، پیش‌خوری‌ها و اتمام کارها را در هر لحظه گام‌به‌گام شرح می‌دهد."
        )
        
        # HTML log rendering
        log_html = "<div style='background-color:#F8FAFC; border: 1px solid #E2E8F0; padding:15px; border-radius:8px; font-family: Courier New, monospace; line-height: 1.6; max-height: 500px; overflow-y: scroll;'>"
        for entry in log:
            log_html += f"<div>{entry}</div>"
        log_html += "</div>"
        
        st.markdown(log_html, unsafe_allow_html=True)
        
    with t_theory:
        st.subheader("Algorithm Analysis & Theoretical Overview" if lang == "English" else "تحلیل تئوری و علمی الگوریتم زمان‌بندی")
        
        if lang == "English":
            st.markdown("""
            ### What is Least Laxity First (LLF)?
            The **Least Laxity First (LLF)** or **Least Slack Time First (LST)** algorithm is a dynamic real-time scheduling algorithm. It assigns priority based on the **laxity (slack time)** of tasks.
            
            #### Mathematical Formulation
            Let a task $T_i$ have an absolute deadline $D_i$ and a remaining execution time $C_i(t)$ at time $t$. The laxity $L_i(t)$ is defined as:
            $$L_i(t) = D_i - t - C_i(t)$$
            
            At any scheduling point:
            * The task with the **smallest laxity** is assigned the highest priority and scheduled on the processor.
            * Laxity is a measure of urgency. If $L_i(t) = 0$, it means task $T_i$ must run immediately to complete by $D_i$. If it waits even one step, it is guaranteed to miss its deadline ($L_i(t) < 0$).
            
            #### Core Properties & Optimality
            1. **Preemptive Optimality**: Preemptive LLF is **optimal** on uniprocessor systems. If a set of tasks can be scheduled by any algorithm to meet all deadlines, LLF is guaranteed to schedule it successfully.
            2. **Dynamic Priority**: Priorities are dynamic because the laxity changes dynamically as time progresses and tasks execute.
            
            #### The Thrashing Limitation (The Laxity Tie Problem)
            A major challenge with preemptive LLF is **thrashing (frequent context switching)**.
            * Consider two tasks $T_1$ and $T_2$ with equal laxity.
            * At time $t$, $T_1$ runs. Its remaining execution time decreases, so its laxity remains constant.
            * Meanwhile, $T_2$ waits. Its remaining execution time is unchanged, so its laxity decreases by 1.
            * At $t+1$, $T_2$ has smaller laxity than $T_1$ and preempts $T_1$.
            * Now $T_2$ runs and its laxity remains constant, while $T_1$ waits and its laxity decreases.
            * At $t+2$, $T_1$ now has smaller laxity and preempts $T_2$.
            * This leads to a context switch at **every single time step**!
            
            #### Solutions to Thrashing (Selectable in this app):
            * **Keep Currently Running (Tie-Breaker)**: If two tasks have equal laxity, priority is given to the one currently running. This breaks the thrashing cycle and avoids unnecessary context switches, making LLF practical for real embedded systems.
            """)
        else:
            st.markdown("""
            ### الگوریتم زمان‌بندی کمترین سستی (LLF) چیست؟
            الگوریتم **Least Laxity First (LLF)** که با نام **Least Slack Time First (LST)** نیز شناخته می‌شود، یک الگوریتم زمان‌بندی پویا برای سیستم‌های زمان‌واقعی (Real-Time) سخت است. در این الگوریتم، اولویت وظایف بر اساس میزان **سستی (Laxity یا Slack)** تعیین می‌شود.
            
            #### فرمول ریاضی سستی
            برای وظیفه $T_i$ با ضرب‌الاجل مطلق $D_i$ و زمان باقیمانده اجرای محاسبات $C_i(t)$ در لحظه $t$، سستی $L_i(t)$ برابر است با:
            $$L_i(t) = D_i - t - C_i(t)$$
            
            مکانیزم تصمیم‌گیری زمان‌بند:
            * وظیفه‌ای که **کمترین سستی** را دارد، بالاترین اولویت را دریافت کرده و پردازنده به آن اختصاص می‌یابد.
            * سستی معیاری از فوریت کار است. اگر $L_i(t) = 0$ شود، وظیفه $T_i$ باید دقیقاً از همان لحظه اجرا شود تا ضرب‌الاجل را از دست ندهد. اگر حتی یک گام منتظر بماند، سستی منفی شده و قطعاً از ضرب‌الاجل عبور خواهد کرد ($L_i(t) < 0$).
            
            #### ویژگی‌های اصلی و بهینگی (Optimality)
            ۱. **بهینگی پیش‌خورانه**: الگوریتم LLF پیش‌خورانه روی سیستم‌های تک‌پردازنده‌ای **بهینه** است. یعنی اگر مجموعه‌ای از کارها توسط هر الگوریتم دیگری قابل زمان‌بندی باشد، LLF نیز قطعاً می‌تواند آنها را بدون از دست رفتن ضرب‌الاجل زمان‌بندی کند.
            ۲. **اولویت پویا**: اولویت‌ها در طول زمان تغییر می‌کنند، زیرا زمان جلو می‌رود و بسته به اجرا شدن یا نشدن وظایف، سستی آنها تغییر می‌کند.
            
            #### چالش لرزش یا تعویض قالب شدید (Thrashing)
            بزرگترین چالش در الگوریتم LLF سنتی، پدیده **لرزش (Thrashing)** است:
            * فرض کنید دو وظیفه $T_1$ و $T_2$ سستی کاملاً برابری داشته باشند.
            * در لحظه $t$ وظیفه $T_1$ اجرا می‌شود. چون اجرا شده سستی آن ثابت می‌ماند. اما $T_2$ متوقف است و سستی آن ۱ واحد کاهش می‌یابد.
            * در لحظه $t+1$ وظیفه $T_2$ سستی کمتری پیدا کرده و جایگزین $T_1$ می‌شود (پیش‌خوری).
            * در این لحظه $T_2$ اجرا می‌شود و سستی‌اش ثابت می‌ماند، اما $T_1$ متوقف مانده و سستی‌اش کاهش می‌یابد.
            * در لحظه $t+2$ مجدداً $T_1$ سستی کمتری پیدا کرده و جایگزین $T_2$ می‌شود.
            * این فرآیند باعث می‌شود در **هر گام زمانی تک‌واحدی** یک تعویض قالب (Context Switch) رخ دهد که سربار پردازشی بسیار شدیدی به سیستم تحمیل می‌کند.
            
            #### راه‌حل‌های حل مشکل لرزش (تعبیه شده در این برنامه):
            * **سیاست حفظ وظیفه در حال اجرا (Keep Currently Running)**: در زمان تساوی سستی، پردازنده اولویت را به کاری می‌دهد که در گام قبلی در حال اجرا بوده است. این سیاست ساده مانع از لرزش مداوم بین کارهای با سستی یکسان شده و تعداد تعویض‌های قالب را به شدت کاهش می‌دهد، که برای سیستم‌های جاسازی‌شده واقعی ضروری است.
            """)
            
else:
    st.warning("Please define a set of tasks." if lang == "English" else "لطفاً مجموعه‌ای از وظایف را تعریف کنید.")
