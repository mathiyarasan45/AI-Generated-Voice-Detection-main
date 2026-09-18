document.addEventListener('DOMContentLoaded', () => {
    const audioFileInput = document.getElementById('audioFile');
    const dropzone = document.getElementById('dropzone');
    const fileLabel = document.getElementById('fileLabel');
    const fileMeta = document.getElementById('fileMeta');
    const fileNameDisplay = document.getElementById('fileName');
    const clearFileBtn = document.getElementById('clearFileBtn');
    
    const recordBtn = document.getElementById('recordBtn');
    const recordBtnText = document.getElementById('recordBtnText');
    const recordDot = document.getElementById('recordDot');
    const micIcon = document.getElementById('micIcon');
    
    const detectBtn = document.getElementById('detectBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const languageSelect = document.getElementById('languageSelect');
    
    const resultContainer = document.getElementById('resultContainer');
    const classificationBadge = document.getElementById('classificationBadge');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceBar = document.getElementById('confidenceBar');
    const explanationText = document.getElementById('explanationText');
    const errorBanner = document.getElementById('errorBanner');
    const errorText = document.getElementById('errorText');

    const API_URL = '/api/voice-detection';
    const API_KEY = 'test_key_123';

    let selectedFile = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let recordingTimer = null;
    let secondsElapsed = 0;

    // Prevent default drag behaviors on window to prevent browser from navigating/opening dropped file
    ['dragover', 'drop'].forEach(eventName => {
        window.addEventListener(eventName, (e) => {
            e.preventDefault();
        });
    });

    // Drag & Drop visual feedback & file drop handling
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileSelect(dt.files[0]);
        }
    });

    dropzone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            audioFileInput.click();
        }
    });

    // File Input Selection
    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        const allowedExtensions = ['.wav', '.mp3', '.aac', '.m4a', '.ogg', '.webm', '.mpeg'];
        const fileName = file.name ? file.name.toLowerCase() : '';
        const isValid = allowedExtensions.some(ext => fileName.endsWith(ext)) || (file.type && (file.type.startsWith('audio/') || file.type.startsWith('video/')));

        if (!isValid) {
            resetFileSelection();
            showError('Invalid file format. Please select a supported audio file (.wav, .mp3, .aac, .m4a, .ogg, .webm, .mpeg).');
            return;
        }

        selectedFile = file;
        fileNameDisplay.textContent = file.name;
        fileMeta.classList.remove('hidden');
        fileLabel.textContent = 'Selected Audio File:';
        detectBtn.disabled = false;
        hideError();
        resultContainer.classList.add('hidden');
    }

    // Clear File Selection
    clearFileBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetFileSelection();
    });

    function resetFileSelection() {
        selectedFile = null;
        audioFileInput.value = '';
        fileMeta.classList.add('hidden');
        fileLabel.textContent = 'Click or drag an audio file (WAV, MP3, AAC, M4A, OGG, WEBM, MPEG)';
        detectBtn.disabled = true;
        resultContainer.classList.add('hidden');
        hideError();
    }

    // Record Voice Button Handler
    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        hideError();
        audioChunks = [];

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(track => track.stop());
                clearInterval(recordingTimer);

                const webmBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
                
                try {
                    const wavBlob = await convertBlobToWav(webmBlob);
                    const recordedFile = new File([wavBlob], 'Recorded_Voice.wav', { type: 'audio/wav' });
                    handleFileSelect(recordedFile);
                    fileLabel.textContent = `Recorded Voice (${secondsElapsed}s):`;
                } catch (convErr) {
                    showError('Failed to process recorded audio: ' + convErr.message);
                }

                resetRecordBtnUI();
            };

            mediaRecorder.start();
            isRecording = true;
            secondsElapsed = 0;

            recordBtn.classList.add('recording');
            recordDot.classList.remove('hidden');
            micIcon.classList.add('hidden');
            recordBtnText.textContent = 'Stop Recording (00:00)';

            recordingTimer = setInterval(() => {
                secondsElapsed++;
                const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
                const secs = String(secondsElapsed % 60).padStart(2, '0');
                recordBtnText.textContent = `Stop Recording (${mins}:${secs})`;
            }, 1000);

        } catch (err) {
            showError('Microphone access denied or not available: ' + err.message);
            resetRecordBtnUI();
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
        }
    }

    function resetRecordBtnUI() {
        isRecording = false;
        clearInterval(recordingTimer);
        recordBtn.classList.remove('recording');
        recordDot.classList.add('hidden');
        micIcon.classList.remove('hidden');
        recordBtnText.textContent = 'Record Voice';
    }

    // Convert audio Blob to 16kHz WAV Blob
    async function convertBlobToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        try {
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            return audioBufferToWavBlob(audioBuffer);
        } finally {
            if (audioCtx.state !== 'closed') {
                audioCtx.close().catch(() => {});
            }
        }
    }

    function audioBufferToWavBlob(buffer) {
        const numOfChan = 1;
        const sampleRate = buffer.sampleRate;
        const channelData = buffer.getChannelData(0);
        const length = channelData.length * 2 + 44;
        const out = new DataView(new ArrayBuffer(length));
        let pos = 0;

        function setUint16(data) { out.setUint16(pos, data, true); pos += 2; }
        function setUint32(data) { out.setUint32(pos, data, true); pos += 4; }

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8);
        setUint32(0x45564157); // "WAVE"
        setUint32(0x20746d66); // "fmt "
        setUint32(16);
        setUint16(1);
        setUint16(numOfChan);
        setUint32(sampleRate);
        setUint32(sampleRate * 2 * numOfChan);
        setUint16(numOfChan * 2);
        setUint16(16);
        setUint32(0x61746164); // "data"
        setUint32(length - pos - 4);

        for (let i = 0; i < channelData.length; i++) {
            let sample = Math.max(-1, Math.min(1, channelData[i]));
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
            out.setInt16(pos, sample, true);
            pos += 2;
        }

        return new Blob([out], { type: 'audio/wav' });
    }

    // Detect Voice Submit Action
    detectBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        setLoading(true);
        hideError();

        try {
            const base64Data = await fileToBase64(selectedFile);
            const fileExt = selectedFile.name.split('.').pop().toLowerCase() || 'wav';

            const payload = {
                language: languageSelect.value,
                audioFormat: fileExt,
                audioBase64: base64Data
            };

            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': API_KEY
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Voice detection failed');
            }

            displayResult(data);

        } catch (err) {
            showError(err.message || 'An unexpected error occurred during voice detection.');
        } finally {
            setLoading(false);
        }
    });

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const result = reader.result;
                const base64 = result.includes(',') ? result.split(',')[1] : result;
                resolve(base64);
            };
            reader.onerror = (error) => reject(error);
            reader.readAsDataURL(file);
        });
    }

    function displayResult(data) {
        resultContainer.classList.remove('hidden');

        const classification = data.classification || 'UNKNOWN';
        const confidence = typeof data.confidenceScore === 'number' ? data.confidenceScore : 0;
        const explanation = data.explanation || 'No details provided.';

        if (classification === 'AI_GENERATED') {
            classificationBadge.textContent = 'AI GENERATED';
            classificationBadge.className = 'badge badge-ai';
            confidenceBar.style.background = '#a855f7';
        } else {
            classificationBadge.textContent = 'HUMAN';
            classificationBadge.className = 'badge badge-human';
            confidenceBar.style.background = '#10b981';
        }

        const percentage = Math.round(confidence * 100);
        confidenceText.textContent = `${percentage}%`;
        confidenceBar.style.width = `${percentage}%`;

        explanationText.textContent = explanation;
    }

    function setLoading(isLoading) {
        detectBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = 'Analyzing Audio...';
            btnSpinner.classList.remove('hidden');
        } else {
            btnText.textContent = 'Detect Voice';
            btnSpinner.classList.add('hidden');
        }
    }

    function showError(message) {
        errorText.textContent = message;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        errorBanner.classList.add('hidden');
    }
});
