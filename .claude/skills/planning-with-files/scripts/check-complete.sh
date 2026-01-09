#!/bin/bash
# Check if all phases in a task's task_plan.md are complete
# Usage: ./check-complete.sh [task-name]
# If no task name provided, checks all tasks in docs/tasks/
# Exit 0 if complete, exit 1 if incomplete
# Used by Stop hook to verify task completion

set -e

check_task_plan() {
    local PLAN_FILE="$1"
    local TASK_NAME="$2"
    
    if [ ! -f "$PLAN_FILE" ]; then
        echo "ERROR: $PLAN_FILE not found"
        return 1
    fi
    
    # Count phases by status (using -F for fixed string matching)
    TOTAL=$(grep -c "### Phase" "$PLAN_FILE" || true)
    COMPLETE=$(grep -cF "**Status:** complete" "$PLAN_FILE" || true)
    IN_PROGRESS=$(grep -cF "**Status:** in_progress" "$PLAN_FILE" || true)
    PENDING=$(grep -cF "**Status:** pending" "$PLAN_FILE" || true)
    
    # Default to 0 if empty
    : "${TOTAL:=0}"
    : "${COMPLETE:=0}"
    : "${IN_PROGRESS:=0}"
    : "${PENDING:=0}"
    
    echo "Task: $TASK_NAME"
    echo "  Total phases:   $TOTAL"
    echo "  Complete:       $COMPLETE"
    echo "  In progress:    $IN_PROGRESS"
    echo "  Pending:        $PENDING"
    
    # Return success only if all complete
    if [ "$COMPLETE" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo "  → ALL PHASES COMPLETE ✓"
        return 0
    else
        echo "  → INCOMPLETE"
        return 1
    fi
}

echo "=== Task Completion Check ==="
echo ""

# If specific task provided
if [ -n "$1" ]; then
    TASK_DIR="docs/tasks/$1"
    PLAN_FILE="$TASK_DIR/task_plan.md"
    
    if check_task_plan "$PLAN_FILE" "$1"; then
        echo ""
        echo "TASK COMPLETE"
        exit 0
    else
        echo ""
        echo "TASK NOT COMPLETE"
        echo "Do not stop until all phases are complete."
        exit 1
    fi
fi

# Check all tasks in docs/tasks/
TASKS_DIR="docs/tasks"
if [ ! -d "$TASKS_DIR" ]; then
    echo "No tasks directory found at $TASKS_DIR"
    echo "Create a task folder first with: mkdir -p $TASKS_DIR/<task-name>"
    exit 1
fi

ALL_COMPLETE=true
TASK_COUNT=0

for task_dir in "$TASKS_DIR"/*/; do
    if [ -d "$task_dir" ]; then
        TASK_NAME=$(basename "$task_dir")
        PLAN_FILE="$task_dir/task_plan.md"
        
        if [ -f "$PLAN_FILE" ]; then
            TASK_COUNT=$((TASK_COUNT + 1))
            if ! check_task_plan "$PLAN_FILE" "$TASK_NAME"; then
                ALL_COMPLETE=false
            fi
            echo ""
        fi
    fi
done

if [ "$TASK_COUNT" -eq 0 ]; then
    echo "No task plans found in $TASKS_DIR"
    echo "Create a task with: ./init-session.sh <task-name>"
    exit 1
fi

echo "=== Summary ==="
if $ALL_COMPLETE; then
    echo "ALL TASKS COMPLETE ✓"
    exit 0
else
    echo "TASKS NOT COMPLETE"
    echo "Do not stop until all phases are complete."
    exit 1
fi
