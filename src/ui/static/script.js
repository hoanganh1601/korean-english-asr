// Korean-English ASR Studio Client Logic

let currentAudioBlob = null;
let currentAudioFile = null;
let currentSampleUrl = null;
let samplesData = [];
let debounceTimer = null;

// DOM Elements
const dropzone = document.getElementById("dropzone");
const audioFileInput = document.getElementById("audioFileInput");
const audioControlBar = document.getElementById("audioControlBar");
const audioPlayer = document.getElementById("audioPlayer");
const currentFileName = document.getElementById("currentFileName");
const currentDuration = document.getElementById("currentDuration");
const presetButtons = document.getElementById("presetButtons");
const btnTranscribe = document.getElementById("btnTranscribe");
const btnClear = document.getElementById("btnClear");
const btnSpinner = document.getElementById("btnSpinner");
const transcribeBtnText = document.getElementById("transcribeBtnText");
const modelModeSelect = document.getElementById("modelModeSelect");
const modelLabel = document.getElementById("modelLabel");

const refTextarea = document.getElementById("refTextarea");
const predTextarea = document.getElementById("predTextarea");
const refWordCount = document.getElementById("refWordCount");
const predWordCount = document.getElementById("predWordCount");
const latencyBadge = document.getElementById("latencyBadge");
const btnCopyRef = document.getElementById("btnCopyRef");
const btnCopyPred = document.getElementById("btnCopyPred");
const btnRecalculate = document.getElementById("btnRecalculate");

// Metrics & Errors
const werValue = document.getElementById("werValue");
const werBar = document.getElementById("werBar");
const werStatus = document.getElementById("werStatus");
const cerValue = document.getElementById("cerValue");
const cerBar = document.getElementById("cerBar");
const accValue = document.getElementById("accValue");
const accBar = document.getElementById("accBar");

const cntSubs = document.getElementById("cntSubs");
const cntDels = document.getElementById("cntDels");
const cntIns = document.getElementById("cntIns");
const cntReps = document.getElementById("cntReps");

const badgeCountRep = document.getElementById("badgeCountRep");
const badgeCountSub = document.getElementById("badgeCountSub");
const badgeCountDel = document.getElementById("badgeCountDel");
const badgeCountIns = document.getElementById("badgeCountIns");

const alignmentTokens = document.getElementById("alignmentTokens");
const listRepetitions = document.getElementById("listRepetitions");
const listSubstitutions = document.getElementById("listSubstitutions");
const listDeletions = document.getElementById("listDeletions");
const listInsertions = document.getElementById("listInsertions");
const toast = document.getElementById("toast");


// Initialize
document.addEventListener("DOMContentLoaded", () => {
    initDropzone();
    loadSampleAudios();
    setupEventListeners();
    updateWordCounts();
});

function showToast(msg, duration = 3000) {
    toast.textContent = msg;
    toast.style.display = "block";
    setTimeout(() => {
        toast.style.display = "none";
    }, duration);
}

// Model Selection handler
modelModeSelect.addEventListener("change", (e) => {
    const mode = e.target.value;
    if (mode === "lora_v4") {
        modelLabel.textContent = "BuzzASR + LoRA v4";
    } else if (mode === "lora_v3") {
        modelLabel.textContent = "BuzzASR + LoRA v3";
    } else {
        modelLabel.textContent = "BuzzASR Base";
    }
});

// Dropzone & File Upload
function initDropzone() {
    dropzone.addEventListener("click", () => audioFileInput.click());

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });
}

async function handleSelectedFile(file) {
    currentAudioFile = file;
    currentAudioBlob = file;
    currentSampleUrl = null;

    currentFileName.textContent = file.name;
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioControlBar.style.display = "flex";
    btnTranscribe.disabled = false;

    // Reset active preset button
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));

    // Auto-Lookup Real Text from Manifest / Dataset
    tryLookupRealText(file.name);
}

// Auto Lookup Real Text via API
async function tryLookupRealText(filename) {
    try {
        const res = await fetch(`/api/lookup_text?filename=${encodeURIComponent(filename)}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.found && data.text) {
            refTextarea.value = data.text;
            updateWordCounts();
            showToast(`🎯 Tự động tìm thấy Real Text từ dữ liệu [${data.source}]!`, 3500);
            triggerReactiveEvaluation();
        } else {
            showToast(`Đã nạp file: ${filename}`);
        }
    } catch (e) {
        console.error("Lookup real text error:", e);
    }
}

// Load presets from backend
async function loadSampleAudios() {
    try {
        const res = await fetch("/api/samples");
        if (!res.ok) throw new Error("Failed to fetch samples");
        samplesData = await res.json();

        presetButtons.innerHTML = "";
        samplesData.forEach((sample, idx) => {
            const btn = document.createElement("button");
            btn.className = "preset-btn";
            const isCS = sample.source.includes("HiKE");
            const tagClass = isCS ? "tag-cs" : "tag-kr";
            const tagLabel = isCS ? "Code-Switch" : "Korean";

            btn.innerHTML = `
                <span class="preset-tag ${tagClass}">${tagLabel}</span>
                <span>Mẫu ${idx + 1} (${sample.duration_sec.toFixed(1)}s)</span>
            `;

            btn.addEventListener("click", () => selectPreset(sample, btn));
            presetButtons.appendChild(btn);
        });
    } catch (err) {
        console.error("Error loading sample audios:", err);
        presetButtons.innerHTML = `<span class="placeholder-text">Không thể tải danh sách mẫu tự động</span>`;
    }
}

async function selectPreset(sample, btnElement) {
    document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
    btnElement.classList.add("active");

    const fname = sample.file.split(/[\\/]/).pop();
    currentFileName.textContent = fname;
    currentDuration.textContent = `${sample.duration_sec.toFixed(1)} giây • ${sample.source}`;

    currentSampleUrl = sample.url;
    audioPlayer.src = sample.url;
    audioControlBar.style.display = "flex";
    btnTranscribe.disabled = false;

    // Set reference text
    refTextarea.value = sample.reference;
    updateWordCounts();

    // Fetch the sample audio as blob
    try {
        const res = await fetch(sample.url);
        currentAudioBlob = await res.blob();
        currentAudioFile = new File([currentAudioBlob], fname, { type: "audio/wav" });
    } catch (e) {
        console.error("Failed to load sample blob:", e);
    }

    showToast(`Đã chọn: ${fname}`);
}

// Setup event listeners
function setupEventListeners() {
    btnTranscribe.addEventListener("click", runTranscription);

    btnClear.addEventListener("click", () => {
        refTextarea.value = "";
        predTextarea.value = "";
        currentAudioBlob = null;
        currentAudioFile = null;
        audioPlayer.src = "";
        audioControlBar.style.display = "none";
        btnTranscribe.disabled = true;
        latencyBadge.style.display = "none";
        document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        updateWordCounts();
        resetDiagnostics();
        showToast("Đã xóa tất cả dữ liệu");
    });

    btnCopyRef.addEventListener("click", () => copyToClipboard(refTextarea.value, "Đã sao chép Text Gốc!"));
    btnCopyPred.addEventListener("click", () => copyToClipboard(predTextarea.value, "Đã sao chép Prediction!"));

    refTextarea.addEventListener("input", () => {
        updateWordCounts();
        triggerReactiveEvaluation();
    });

    predTextarea.addEventListener("input", () => {
        updateWordCounts();
        triggerReactiveEvaluation();
    });

    btnRecalculate.addEventListener("click", triggerEvaluationNow);

    audioPlayer.addEventListener("loadedmetadata", () => {
        if (audioPlayer.duration && !currentSampleUrl) {
            currentDuration.textContent = `${audioPlayer.duration.toFixed(1)} giây`;
        }
    });
}

function copyToClipboard(text, successMsg) {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => showToast(successMsg));
}

function updateWordCounts() {
    const refCount = refTextarea.value.trim() ? refTextarea.value.trim().split(/\s+/).length : 0;
    const predCount = predTextarea.value.trim() ? predTextarea.value.trim().split(/\s+/).length : 0;
    refWordCount.textContent = `${refCount} từ`;
    predWordCount.textContent = `${predCount} từ`;
}

function triggerReactiveEvaluation() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        triggerEvaluationNow();
    }, 400);
}

// Fast Transcription Flow
async function runTranscription() {
    if (!currentAudioBlob) {
        showToast("Vui lòng đẩy file audio lên trước!");
        return;
    }

    btnTranscribe.disabled = true;
    btnSpinner.style.display = "block";
    transcribeBtnText.textContent = "⚡ Mô hình ASR đang nhận dạng...";

    const formData = new FormData();
    const filename = currentAudioFile ? currentAudioFile.name : "audio.wav";
    formData.append("audio", currentAudioBlob, filename);
    formData.append("reference", refTextarea.value.trim());
    formData.append("model_mode", modelModeSelect.value);

    const startTime = performance.now();

    try {
        const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Lỗi xử lý âm thanh");
        }

        const data = await response.json();
        predTextarea.value = data.transcription;

        // Auto populate Real Text if returned from server lookup and textarea was empty
        if (!refTextarea.value.trim() && data.reference_text) {
            refTextarea.value = data.reference_text;
            showToast("🎯 Đã tự động khớp Real Text từ manifest của file!");
        }

        updateWordCounts();

        // Update latency badge
        const latencyMs = Math.round(data.inference_time_sec * 1000);
        latencyBadge.textContent = `${latencyMs} ms (RTF: ${data.real_time_factor.toFixed(2)})`;
        latencyBadge.style.display = "inline-block";

        if (data.analysis) {
            renderDiagnostics(data.analysis);
        } else {
            triggerEvaluationNow();
        }

        const totalClientMs = Math.round(performance.now() - startTime);
        showToast(`Nhận dạng xong trong ${latencyMs} ms (Tổng: ${totalClientMs} ms)!`);
    } catch (err) {
        console.error("Transcription error:", err);
        showToast(`Lỗi: ${err.message}`);
    } finally {
        btnTranscribe.disabled = false;
        btnSpinner.style.display = "none";
        transcribeBtnText.textContent = "Chuyển Audio Sang Text & So Sánh";
    }
}

// Evaluation Flow
async function triggerEvaluationNow() {
    const ref = refTextarea.value.trim();
    const pred = predTextarea.value.trim();

    if (!ref && !pred) {
        resetDiagnostics();
        return;
    }

    try {
        const res = await fetch("/api/evaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reference: ref, prediction: pred })
        });
        if (!res.ok) throw new Error("Evaluation request failed");
        const data = await res.json();
        renderDiagnostics(data);
    } catch (err) {
        console.error("Evaluation error:", err);
    }
}

function resetDiagnostics() {
    werValue.textContent = "0.00%";
    werBar.style.width = "0%";
    werStatus.textContent = "Chưa có dữ liệu";
    cerValue.textContent = "0.00%";
    cerBar.style.width = "0%";
    accValue.textContent = "100.00%";
    accBar.style.width = "100%";

    cntSubs.textContent = "0";
    cntDels.textContent = "0";
    cntIns.textContent = "0";
    cntReps.textContent = "0";

    badgeCountRep.textContent = "0";
    badgeCountSub.textContent = "0";
    badgeCountDel.textContent = "0";
    badgeCountIns.textContent = "0";

    alignmentTokens.innerHTML = `<p class="placeholder-text">Chưa có dữ liệu so khớp. Hãy nhập hoặc chuyển audio để xem.</p>`;
    listRepetitions.innerHTML = `<div class="empty-state">✅ Không phát hiện từ bị lặp hoặc trùng lặp</div>`;
    listSubstitutions.innerHTML = `<div class="empty-state">✅ Không có từ nào bị sai</div>`;
    listDeletions.innerHTML = `<div class="empty-state">✅ Không có từ nào bị thiếu</div>`;
    listInsertions.innerHTML = `<div class="empty-state">✅ Không có từ nào bị thừa</div>`;
}

function renderDiagnostics(data) {
    // 1. WER
    const wer = data.wer;
    werValue.textContent = `${wer.toFixed(2)}%`;
    werBar.style.width = `${Math.min(100, wer)}%`;

    if (wer <= 10.0) {
        werValue.style.color = "#34d399";
        werBar.style.backgroundColor = "#10b981";
        werStatus.textContent = "Xuất sắc (< 10%)";
    } else if (wer <= 25.0) {
        werValue.style.color = "#38bdf8";
        werBar.style.backgroundColor = "#06b6d4";
        werStatus.textContent = "Rất tốt (10% - 25%)";
    } else if (wer <= 50.0) {
        werValue.style.color = "#fbbf24";
        werBar.style.backgroundColor = "#f59e0b";
        werStatus.textContent = "Trung bình (25% - 50%)";
    } else {
        werValue.style.color = "#f43f5e";
        werBar.style.backgroundColor = "#ef4444";
        werStatus.textContent = "Cao (> 50%)";
    }

    // 2. CER
    const cer = data.cer;
    cerValue.textContent = `${cer.toFixed(2)}%`;
    cerBar.style.width = `${Math.min(100, cer)}%`;

    // 3. Accuracy
    const acc = data.accuracy;
    accValue.textContent = `${acc.toFixed(2)}%`;
    accBar.style.width = `${acc}%`;

    // Counts
    cntSubs.textContent = data.counts.substitutions;
    cntDels.textContent = data.counts.deletions;
    cntIns.textContent = data.counts.insertions;
    cntReps.textContent = data.counts.repetitions;

    badgeCountRep.textContent = data.counts.repetitions;
    badgeCountSub.textContent = data.counts.substitutions;
    badgeCountDel.textContent = data.counts.deletions;
    badgeCountIns.textContent = data.counts.insertions;

    // Visual Alignment Diff
    renderAlignmentTokens(data.visual_alignment);

    // Error Lists
    renderRepetitions(data.repetitions);
    renderSubstitutions(data.substitutions);
    renderDeletions(data.deletions);
    renderInsertions(data.insertions);
}

function renderAlignmentTokens(alignment) {
    if (!alignment || alignment.length === 0) {
        alignmentTokens.innerHTML = `<p class="placeholder-text">Không có token so khớp</p>`;
        return;
    }

    alignmentTokens.innerHTML = "";
    alignment.forEach(tok => {
        const span = document.createElement("span");
        span.className = `token-pill token-${tok.type}`;

        if (tok.type === "equal") {
            span.textContent = tok.pred || tok.ref;
        } else if (tok.type === "substitute") {
            span.innerHTML = `
                <span class="token-ref">${tok.ref}</span>
                <span class="token-arrow">➔</span>
                <span class="token-pred">${tok.pred}</span>
            `;
        } else if (tok.type === "delete") {
            span.innerHTML = `<span>${tok.ref}</span>`;
            span.title = "Từ bị thiếu (bỏ sót)";
        } else if (tok.type === "insert") {
            span.innerHTML = `<span>+${tok.pred}</span>`;
            span.title = "Từ bị thừa (chèn thêm)";
        }
        alignmentTokens.appendChild(span);
    });
}

function renderRepetitions(repetitions) {
    if (!repetitions || repetitions.length === 0) {
        listRepetitions.innerHTML = `<div class="empty-state">✅ Không phát hiện từ bị lặp hoặc trùng lặp</div>`;
        return;
    }

    listRepetitions.innerHTML = "";
    repetitions.forEach(item => {
        const div = document.createElement("div");
        div.className = "error-item";
        div.innerHTML = `
            <div class="error-item-content">
                <strong>"${item.word}"</strong>
                <span class="error-badge-tag">${item.type}</span>
            </div>
            <span class="pill-rep" style="padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">${item.count} lần</span>
        `;
        listRepetitions.appendChild(div);
    });
}

function renderSubstitutions(subs) {
    if (!subs || subs.length === 0) {
        listSubstitutions.innerHTML = `<div class="empty-state">✅ Không có từ nào bị sai</div>`;
        return;
    }

    listSubstitutions.innerHTML = "";
    subs.forEach(item => {
        const div = document.createElement("div");
        div.className = "error-item";
        div.innerHTML = `
            <div class="error-item-content">
                <span style="color: #fda4af; text-decoration: line-through;">${item.ref}</span>
                <span class="error-arrow">➔</span>
                <span style="color: #fde047; font-weight: 600;">${item.pred}</span>
            </div>
            <span class="pill-sub" style="padding: 2px 8px; border-radius: 4px; font-size: 0.72rem;">Sai lệch</span>
        `;
        listSubstitutions.appendChild(div);
    });
}

function renderDeletions(dels) {
    if (!dels || dels.length === 0) {
        listDeletions.innerHTML = `<div class="empty-state">✅ Không có từ nào bị thiếu</div>`;
        return;
    }

    listDeletions.innerHTML = "";
    dels.forEach(item => {
        const div = document.createElement("div");
        div.className = "error-item";
        div.innerHTML = `
            <div class="error-item-content">
                <span style="color: #fb7185; font-weight: 600;">"${item.ref}"</span>
            </div>
            <span class="pill-del" style="padding: 2px 8px; border-radius: 4px; font-size: 0.72rem;">Bị thiếu trong audio</span>
        `;
        listDeletions.appendChild(div);
    });
}

function renderInsertions(ins) {
    if (!ins || ins.length === 0) {
        listInsertions.innerHTML = `<div class="empty-state">✅ Không có từ nào bị thừa</div>`;
        return;
    }

    listInsertions.innerHTML = "";
    ins.forEach(item => {
        const div = document.createElement("div");
        div.className = "error-item";
        div.innerHTML = `
            <div class="error-item-content">
                <span style="color: #38bdf8; font-weight: 600;">+ "${item.pred}"</span>
            </div>
            <span class="pill-ins" style="padding: 2px 8px; border-radius: 4px; font-size: 0.72rem;">Dư thừa trong prediction</span>
        `;
        listInsertions.appendChild(div);
    });
}
