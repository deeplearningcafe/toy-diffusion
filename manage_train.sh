#!/bin/bash

# Configuration
ENV_PATH="$HOME/conda/bin/activate"
ENV_NAME="torch"
LOG_DIR="results/logs"
PID_FILE="train.pid"
GPUS="3"

mkdir -p "$LOG_DIR"

get_latest_log() {
    # Finds the most recently modified log file in the log directory
    ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -n 1
}

start_training() {
    if [ -f "$PID_FILE" ]; then
        echo "Training is already running (PID: $(cat $PID_FILE))"
        exit 1
    fi

    # Generate filename only at the moment of starting
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

    echo "Starting training in background..."
    echo "Logs will be saved to: $LOG_FILE"

    export CUDA_VISIBLE_DEVICES=$GPUS
    
    # setsid creates a new session so the process survives logout
    setsid bash -c "
        source $ENV_PATH $ENV_NAME
        python scripts/anime_faces.py
    " > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Training started. PID $! saved to $PID_FILE."
}

stop_training() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Checking for orphan processes..."
        pkill -f "scripts/train.py"
        return
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping process group for PID $PID..."
    
    # Kill the entire process group (negative PID)
    kill -TERM -"$PID" 2>/dev/null
    rm "$PID_FILE"
    
    # Ensure no workers are left behind
    pkill -f "scripts/anime_faces.py"
    echo "Training stopped."
}

tail_logs() {
    LATEST_LOG=$(get_latest_log)
    if [ -z "$LATEST_LOG" ]; then
        echo "No log files found in $LOG_DIR"
        exit 1
    fi
    echo "Tailing latest log: $LATEST_LOG"
    tail -f "$LATEST_LOG"
}

case "$1" in
    start) start_training ;;
    stop)  stop_training ;;
    tail)  tail_logs ;;
    *)     echo "Usage: $0 {start|stop|tail}" ;;
esac