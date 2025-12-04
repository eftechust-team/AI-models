const API_BASE = '';

let currentJobId = null;
let progressInterval = null;

// DOM elements
const textInput = document.getElementById('textInput');
const thicknessInput = document.getElementById('thicknessInput');
const generateBtn = document.getElementById('generateBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const progressMessage = document.getElementById('progressMessage');
const aiStatus = document.getElementById('aiStatus');
const imageSelectionSection = document.getElementById('imageSelectionSection');
const imageGrid = document.getElementById('imageGrid');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const previewCanvas = document.getElementById('previewCanvas');
const downloadBtn = document.getElementById('downloadBtn');
const reverseBtn = document.getElementById('reverseBtn');
const applyDrawingBtn = document.getElementById('applyDrawingBtn');
const blackPenBtn = document.getElementById('blackPenBtn');
const whitePenBtn = document.getElementById('whitePenBtn');
const brushSize = document.getElementById('brushSize');
const brushSizeValue = document.getElementById('brushSizeValue');
const clearDrawingBtn = document.getElementById('clearDrawingBtn');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const inputModeRadios = document.querySelectorAll('input[name="inputMode"]');
const aiInputSection = document.getElementById('aiInputSection');
const uploadInputSection = document.getElementById('uploadInputSection');
const imageUpload = document.getElementById('imageUpload');
const uploadPreview = document.getElementById('uploadPreview');
const uploadPreviewImage = document.getElementById('uploadPreviewImage');
const clearUploadBtn = document.getElementById('clearUploadBtn');
const saveModelBtn = document.getElementById('saveModelBtn');
const savedModelsSection = document.getElementById('savedModelsSection');
const savedModelsGrid = document.getElementById('savedModelsGrid');
const refreshModelsBtn = document.getElementById('refreshModelsBtn');
const noModelsMessage = document.getElementById('noModelsMessage');
const saveModelModal = document.getElementById('saveModelModal');
const modelNameInput = document.getElementById('modelNameInput');
const modelDescriptionInput = document.getElementById('modelDescriptionInput');
const confirmSaveBtn = document.getElementById('confirmSaveBtn');
const cancelSaveBtn = document.getElementById('cancelSaveBtn');

// Store current model data for saving
let currentStlPath = null;
let currentPreviewImage = null;
let currentStlPaths = null; // Store current STL paths for multi-layer mode
let currentLayerInfo = null; // Store current layer info for multi-layer mode

// Store current image data for reversal
let currentImageBase64 = null;
let originalImageBase64 = null; // Store original before any reversal
let isReversed = false; // Track if image is currently reversed

// Drawing state
let isDrawing = false;
let currentPenColor = 'black';
let canvasContext = null;
let canvasOriginalImageData = null; // Store original canvas image data for clearing
let drawingHistory = []; // Array of ImageData for undo/redo
let drawingHistoryIndex = -1; // Current position in history

// Helper function to get solid_infill option
function getSolidInfill() {
    const fillOption = document.querySelector('input[name="fillOption"]:checked');
    return fillOption ? fillOption.value === 'solid' : true; // Default to true
}

// Helper function to get arc_top option
function getArcTop() {
    const arcTopCheckbox = document.getElementById('arcTopCheckbox');
    return arcTopCheckbox ? arcTopCheckbox.checked : false; // Default to false
}

// Helper function to get multi_layer option
function getMultiLayer() {
    const multiLayerCheckbox = document.getElementById('multiLayerCheckbox');
    return multiLayerCheckbox ? multiLayerCheckbox.checked : false; // Default to false
}

// Event listeners
generateBtn.addEventListener('click', handleGenerate);
downloadBtn.addEventListener('click', handleDownload);
reverseBtn.addEventListener('click', handleReverse);

// Mode switching
inputModeRadios.forEach(radio => {
    radio.addEventListener('change', handleModeChange);
});

// Image upload handling
imageUpload.addEventListener('change', handleImageUpload);
clearUploadBtn.addEventListener('click', clearUploadedImage);

// Drawing tool event listeners
if (applyDrawingBtn) {
    applyDrawingBtn.addEventListener('click', handleApplyDrawing);
}
if (blackPenBtn) {
    blackPenBtn.addEventListener('click', () => selectPenColor('black'));
}
if (whitePenBtn) {
    whitePenBtn.addEventListener('click', () => selectPenColor('white'));
}
if (brushSize) {
    brushSize.addEventListener('input', (e) => {
        if (brushSizeValue) {
            brushSizeValue.textContent = e.target.value;
        }
    });
}
if (clearDrawingBtn) {
    clearDrawingBtn.addEventListener('click', clearDrawing);
}
if (undoBtn) {
    undoBtn.addEventListener('click', undoDrawing);
}
if (redoBtn) {
    redoBtn.addEventListener('click', redoDrawing);
}

// Initialize canvas when DOM is ready
if (previewCanvas && previewCanvas.getContext) {
    canvasContext = previewCanvas.getContext('2d');
    setupCanvasDrawing();
    
    // Set initial pen color
    if (blackPenBtn && whitePenBtn) {
        selectPenColor('black');
    }
}

function handleModeChange() {
    const selectedMode = document.querySelector('input[name="inputMode"]:checked').value;
    if (selectedMode === 'ai') {
        aiInputSection.classList.remove('hidden');
        uploadInputSection.classList.add('hidden');
        generateBtn.textContent = 'Generate STL';
    } else {
        aiInputSection.classList.add('hidden');
        uploadInputSection.classList.remove('hidden');
        generateBtn.textContent = 'Generate STL from Image';
    }
    hideError();
    hidePreview();
    hideImageSelection();
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImage.src = e.target.result;
            uploadPreview.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

function clearUploadedImage() {
    imageUpload.value = '';
    uploadPreview.classList.add('hidden');
    uploadPreviewImage.src = '';
}

let eventSource = null;
let receivedImages = [];

function handleGenerate() {
    const selectedMode = document.querySelector('input[name="inputMode"]:checked').value;
    
    if (selectedMode === 'upload') {
        handleUploadGenerate();
    } else {
        handleAIGenerate();
    }
}

function handleUploadGenerate() {
    const file = imageUpload.files[0];
    
    if (!file) {
        showError('Please select an image file to upload');
        return;
    }

    const thickness = parseFloat(thicknessInput.value);
    
    if (isNaN(thickness) || thickness <= 0) {
        showError('Please enter a valid thickness (greater than 0)');
        return;
    }

    const description = document.getElementById('uploadDescription').value.trim();

    // Reset UI
    hideError();
    hidePreview();
    hideImageSelection();
    showProgress();
    updateProgress(0, '');
    generateBtn.disabled = true;
    generateBtn.textContent = 'Processing...';

    // Read the image file and convert to base64
    const reader = new FileReader();
    reader.onload = (e) => {
        const imageDataUrl = e.target.result;
        // Extract base64 data (remove data:image/png;base64, prefix)
        const base64Data = imageDataUrl.split(',')[1];

        // Send to server for processing
        updateProgress(10, '');
        fetch(`${API_BASE}/api/upload-image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: base64Data,
                description: description,
                thickness: thickness,
                solid_infill: getSolidInfill(),
                arc_top: getArcTop(),
                multi_layer: getMultiLayer()
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                resetGenerateButton();
                return;
            }

            if (data.job_id) {
                currentJobId = data.job_id;
                updateProgress(100, '');
                
                // Show preview image if available
                if (data.preview_image) {
                    // Store original image (before any reversal)
                    originalImageBase64 = data.preview_image;
                    currentImageBase64 = data.preview_image;
                    isReversed = false; // Reset reversal state
                    
                    // Load image to canvas for drawing
                    loadImageToCanvas(`data:image/png;base64,${data.preview_image}`);
                    previewImage.src = `data:image/png;base64,${data.preview_image}`;
                    previewSection.classList.remove('hidden');
                }
                
                // Handle STL response (single or multi-layer)
                handleStlResponse(data);
                
                // Hide progress and show preview
                hideProgress();
                resetGenerateButton();
            } else {
                showError('Failed to process image');
                resetGenerateButton();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Failed to process image: ' + error.message);
            resetGenerateButton();
        });
    };
    reader.onerror = () => {
        showError('Failed to read image file');
        resetGenerateButton();
    };
    reader.readAsDataURL(file);
}

function handleAIGenerate() {
    const text = textInput.value.trim();

    if (!text) {
        showError('Please enter a shape description');
        return;
    }

    // Reset UI
    hideError();
    hidePreview();
    hideImageSelection();
    showProgress();
    updateProgress(0, '');
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';
    
    // Clear previous images
    receivedImages = [];
    imageGrid.innerHTML = '';
    
    // Show image selection section (will be populated as images arrive)
    imageSelectionSection.classList.remove('hidden');

    // Use Server-Sent Events to receive images as they're generated
    // First, we need to send the request and get the stream
    fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            num_images: 4
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to start image generation');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    return;
                }
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.type === 'progress') {
                                updateProgress(data.progress, '');
                                currentJobId = data.job_id;
                            } else if (data.type === 'image') {
                                // Add image immediately
                                receivedImages.push(data.image);
                                addImageToGrid(data.image, data.index);
                                updateProgress(data.progress, '');
                            } else if (data.type === 'complete') {
                                updateProgress(100, '');
                                hideProgress();
                                resetGenerateButton();
                            } else if (data.type === 'error') {
                                showError(data.error || 'Failed to generate images');
                                resetGenerateButton();
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e);
                        }
                    }
                }
                
                readStream(); // Continue reading
            }).catch(error => {
                console.error('Stream reading error:', error);
                showError('Failed to generate images: ' + error.message);
                resetGenerateButton();
            });
        }
        
        readStream();
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Failed to generate images: ' + error.message);
        resetGenerateButton();
    });
}

function addImageToGrid(imageBase64, index) {
    const imageOption = document.createElement('div');
    imageOption.className = 'image-option';
    imageOption.dataset.imageIndex = index;
    imageOption.dataset.imageBase64 = imageBase64;
    
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${imageBase64}`;
    img.alt = `Option ${index + 1}`;
    
    imageOption.appendChild(img);
    imageOption.addEventListener('click', () => selectImage(imageOption, imageBase64));
    
    imageGrid.appendChild(imageOption);
}

function showImageSelection(images) {
    // This function is now handled by addImageToGrid as images arrive
    // Keep it for compatibility but it's mainly used for the initial display
    imageGrid.innerHTML = '';
    
    images.forEach((imageBase64, index) => {
        addImageToGrid(imageBase64, index);
    });
    
    imageSelectionSection.classList.remove('hidden');
}

function selectImage(imageOption, imageBase64) {
    // Remove selection from all options
    document.querySelectorAll('.image-option').forEach(opt => {
        opt.classList.remove('selected');
        const checkmark = opt.querySelector('.checkmark');
        if (checkmark) checkmark.remove();
    });
    
    // Add selection to clicked option
    imageOption.classList.add('selected');
    const checkmark = document.createElement('div');
    checkmark.className = 'checkmark';
    checkmark.textContent = '✓';
    imageOption.appendChild(checkmark);
    
    // Generate STL from selected image
    const thickness = parseFloat(thicknessInput.value);
    generateStlFromImage(imageBase64, thickness);
}

function generateStlFromImage(imageBase64, thickness) {
    hideImageSelection();
    showProgress();
    updateProgress(0, '');
    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating STL...';
    
    fetch(`${API_BASE}/api/generate-stl`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
            body: JSON.stringify({
                image: imageBase64,
                thickness: thickness,
                solid_infill: getSolidInfill(),
                arc_top: getArcTop(),
                multi_layer: getMultiLayer()
            })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
            resetGenerateButton();
            return;
        }

        if (data.job_id) {
            currentJobId = data.job_id;
            
            // Show preview image if available
            if (data.preview_image) {
                // Store original image (before any reversal)
                originalImageBase64 = data.preview_image;
                currentImageBase64 = data.preview_image;
                isReversed = false; // Reset reversal state
                
                // Load image to canvas for drawing
                loadImageToCanvas(`data:image/png;base64,${data.preview_image}`);
                previewImage.src = `data:image/png;base64,${data.preview_image}`;
            }
            
            // Store STL path for download
            // Handle STL response (single or multi-layer)
            handleStlResponse(data);
            if (saveModelBtn) saveModelBtn.classList.remove('hidden');
            updateReverseButtonText();
            
            // Start polling for progress
            startProgressPolling();
        } else {
            showError('Failed to generate STL');
            resetGenerateButton();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Failed to generate STL: ' + error.message);
        resetGenerateButton();
    });
}


function startProgressPolling() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }

    progressInterval = setInterval(() => {
        if (!currentJobId) return;

        fetch(`${API_BASE}/api/progress/${currentJobId}`)
            .then(response => response.json())
            .then(data => {
                if (data.progress !== undefined) {
                    updateProgress(data.progress, data.message || 'Processing...');

                    if (data.progress >= 100) {
                        clearInterval(progressInterval);
                        progressInterval = null;
                        onGenerationComplete();
                    }
                }
            })
            .catch(error => {
                console.error('Progress polling error:', error);
            });
    }, 500); // Poll every 500ms
}

function updateProgress(percentage, message) {
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${Math.round(percentage)}%`;
    // Hide progress message - don't show any messages
    progressMessage.textContent = "";
    aiStatus.classList.add('hidden');
}

function onGenerationComplete() {
    updateProgress(100, '');
    setTimeout(() => {
        hideProgress();
        showPreview();
        resetGenerateButton();
        // Note: currentImageBase64 should be set when preview_image is received
    }, 1000);
}

function handleDownload() {
    const stlPath = downloadBtn.dataset.stlPath;
    if (!stlPath) {
        showError('STL file path not available');
        return;
    }

    // Download the STL file
    window.location.href = `${API_BASE}/api/download/${encodeURIComponent(stlPath)}`;
}

function handleReverse() {
    if (!currentImageBase64) {
        showError('No image available to reverse');
        return;
    }

    const thickness = parseFloat(thicknessInput.value);
    
    if (isNaN(thickness) || thickness <= 0) {
        showError('Please enter a valid thickness (greater than 0)');
        return;
    }

    // Reset UI
    hideError();
    showProgress();
    updateProgress(0, '');
    reverseBtn.disabled = true;
    reverseBtn.textContent = isReversed ? 'Undoing...' : 'Reversing...';

    // If we're undoing (already reversed), restore original directly without inverting
    if (isReversed) {
        // Undo: restore original image directly
        if (!originalImageBase64) {
            showError('Original image not available');
            resetReverseButton();
            return;
        }
        
        updateProgress(10, '');
        // Generate STL from original image (no inversion needed)
        fetch(`${API_BASE}/api/generate-stl`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: originalImageBase64,
                thickness: thickness
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                resetReverseButton();
                return;
            }

            if (data.job_id) {
                updateProgress(100, '');
                
                // Restore original image
                if (data.preview_image) {
                    currentImageBase64 = data.preview_image;
                    isReversed = false; // Reset to not reversed
                    
                    // Reload image to canvas
                    loadImageToCanvas(`data:image/png;base64,${data.preview_image}`);
                    previewImage.src = `data:image/png;base64,${data.preview_image}`;
                }
                
                // Store STL path for download
                // Handle STL response (single or multi-layer)
                handleStlResponse(data);
                
                // Update button text and re-enable button
                updateReverseButtonText();
                resetReverseButton();
                
                // Complete immediately
                setTimeout(() => {
                    hideProgress();
                }, 500);
            } else {
                showError('Failed to restore original image');
                resetReverseButton();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Failed to restore original image: ' + error.message);
            resetReverseButton();
        });
    } else {
        // Reverse: invert current image
        // First reversal: save original before reversing
        if (!originalImageBase64) {
            originalImageBase64 = currentImageBase64;
        }
        
        updateProgress(10, '');
        // Send to server for reversal
        fetch(`${API_BASE}/api/reverse-image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImageBase64,
                thickness: thickness,
                solid_infill: getSolidInfill(),
                arc_top: getArcTop(),
                multi_layer: getMultiLayer()
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                resetReverseButton();
                return;
            }

            if (data.job_id) {
                updateProgress(100, '');
                
                // Update stored image and toggle state
                if (data.preview_image) {
                    currentImageBase64 = data.preview_image;
                    isReversed = true; // Mark as reversed
                    
                    // Reload image to canvas
                    loadImageToCanvas(`data:image/png;base64,${data.preview_image}`);
                    previewImage.src = `data:image/png;base64,${data.preview_image}`;
                }
                
                // Store STL path for download
                // Handle STL response (single or multi-layer)
                handleStlResponse(data);
                
                // Update button text and re-enable button
                updateReverseButtonText();
                resetReverseButton();
                
                // Complete immediately (no polling needed for reverse-image)
                setTimeout(() => {
                    hideProgress();
                }, 500);
            } else {
                showError('Failed to reverse image');
                resetReverseButton();
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Failed to reverse image: ' + error.message);
            resetReverseButton();
        });
    }
}

function resetReverseButton() {
    reverseBtn.disabled = false;
    updateReverseButtonText();
}

function updateReverseButtonText() {
    reverseBtn.textContent = isReversed ? 'Undo Reverse' : 'Reverse Selection';
}

function showProgress() {
    progressSection.classList.remove('hidden');
}

function hideProgress() {
    progressSection.classList.add('hidden');
}

function showPreview() {
    previewSection.classList.remove('hidden');
    downloadBtn.classList.remove('hidden');
    reverseBtn.classList.remove('hidden');
    
    // Ensure canvas is initialized
    if (previewCanvas && !canvasContext) {
        canvasContext = previewCanvas.getContext('2d');
        setupCanvasDrawing();
    }
}

function hidePreview() {
    previewSection.classList.add('hidden');
    downloadBtn.classList.add('hidden');
    reverseBtn.classList.add('hidden');
}

function hideImageSelection() {
    imageSelectionSection.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

function hideError() {
    errorSection.classList.add('hidden');
}

// Drawing functions
function setupCanvasDrawing() {
    if (!canvasContext) return;
    
    previewCanvas.addEventListener('mousedown', startDrawing);
    previewCanvas.addEventListener('mousemove', draw);
    previewCanvas.addEventListener('mouseup', stopDrawing);
    previewCanvas.addEventListener('mouseout', stopDrawing);
    
    // Touch support for mobile
    previewCanvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        previewCanvas.dispatchEvent(mouseEvent);
    });
    
    previewCanvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        previewCanvas.dispatchEvent(mouseEvent);
    });
    
    previewCanvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        previewCanvas.dispatchEvent(mouseEvent);
    });
}

function loadImageToCanvas(imageSrc) {
    if (!canvasContext) return;
    
    const img = new Image();
    img.onload = () => {
        // Set canvas size to match image
        previewCanvas.width = img.width;
        previewCanvas.height = img.height;
        
        // Draw image to canvas
        canvasContext.drawImage(img, 0, 0);
        
        // Save original image data for clearing
        canvasOriginalImageData = canvasContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
        
        // Initialize drawing history with the original image
        drawingHistory = [canvasContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height)];
        drawingHistoryIndex = 0;
        updateUndoRedoButtons();
        
        // Show apply button
        if (applyDrawingBtn) {
            applyDrawingBtn.classList.remove('hidden');
        }
    };
    img.src = imageSrc;
}

function startDrawing(e) {
    isDrawing = true;
    const rect = previewCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    drawAt(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const rect = previewCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    drawAt(x, y);
}

function drawAt(x, y) {
    if (!canvasContext) return;
    
    const brushSizeValue = parseInt(brushSize.value);
    const color = currentPenColor === 'black' ? '#000000' : '#FFFFFF';
    
    canvasContext.fillStyle = color;
    canvasContext.beginPath();
    canvasContext.arc(x, y, brushSizeValue / 2, 0, Math.PI * 2);
    canvasContext.fill();
}

function stopDrawing() {
    if (isDrawing) {
        // Save state when drawing stops
        saveDrawingState();
    }
    isDrawing = false;
}

function saveDrawingState() {
    if (!canvasContext) return;
    
    // Remove any states after current index (when drawing after undo)
    if (drawingHistoryIndex < drawingHistory.length - 1) {
        drawingHistory = drawingHistory.slice(0, drawingHistoryIndex + 1);
    }
    
    // Save current state
    const currentState = canvasContext.getImageData(0, 0, previewCanvas.width, previewCanvas.height);
    drawingHistory.push(currentState);
    drawingHistoryIndex = drawingHistory.length - 1;
    
    // Limit history size to prevent memory issues (keep last 50 states)
    if (drawingHistory.length > 50) {
        drawingHistory.shift();
        drawingHistoryIndex--;
    }
    
    updateUndoRedoButtons();
}

function selectPenColor(color) {
    currentPenColor = color;
    if (blackPenBtn) {
        blackPenBtn.classList.toggle('active', color === 'black');
    }
    if (whitePenBtn) {
        whitePenBtn.classList.toggle('active', color === 'white');
    }
}

function clearDrawing() {
    if (!canvasContext || !canvasOriginalImageData) return;
    
    // Restore original image
    canvasContext.putImageData(canvasOriginalImageData, 0, 0);
    
    // Reset history
    drawingHistory = [canvasOriginalImageData];
    drawingHistoryIndex = 0;
    updateUndoRedoButtons();
}

function undoDrawing() {
    if (!canvasContext || drawingHistoryIndex <= 0) return;
    
    drawingHistoryIndex--;
    canvasContext.putImageData(drawingHistory[drawingHistoryIndex], 0, 0);
    updateUndoRedoButtons();
}

function redoDrawing() {
    if (!canvasContext || drawingHistoryIndex >= drawingHistory.length - 1) return;
    
    drawingHistoryIndex++;
    canvasContext.putImageData(drawingHistory[drawingHistoryIndex], 0, 0);
    updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
    if (undoBtn) {
        undoBtn.disabled = drawingHistoryIndex <= 0;
    }
    if (redoBtn) {
        redoBtn.disabled = drawingHistoryIndex >= drawingHistory.length - 1;
    }
}

function handleApplyDrawing() {
    if (!canvasContext) {
        showError('Canvas not available');
        return;
    }

    const thickness = parseFloat(thicknessInput.value);
    
    if (isNaN(thickness) || thickness <= 0) {
        showError('Please enter a valid thickness (greater than 0)');
        return;
    }

    // Get canvas image as base64
    const canvasData = previewCanvas.toDataURL('image/png');
    const base64Data = canvasData.split(',')[1];

    // Reset UI
    hideError();
    showProgress();
    updateProgress(0, '');
    if (applyDrawingBtn) {
        applyDrawingBtn.disabled = true;
        applyDrawingBtn.textContent = 'Processing...';
    }

    // Send to server for STL generation
    updateProgress(10, '');
    fetch(`${API_BASE}/api/generate-stl`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: base64Data,
            thickness: thickness
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
            resetApplyButton();
            return;
        }

        if (data.job_id) {
            updateProgress(100, '');
            
            // Update stored images
            if (data.preview_image) {
                originalImageBase64 = data.preview_image;
                currentImageBase64 = data.preview_image;
                isReversed = false;
                
                // Reload image to canvas
                loadImageToCanvas(`data:image/png;base64,${data.preview_image}`);
                previewImage.src = `data:image/png;base64,${data.preview_image}`;
            }
            
            // Store STL path for download
            downloadBtn.dataset.stlPath = data.stl_path;
            currentStlPath = data.stl_path;
            if (data.preview_image) currentPreviewImage = data.preview_image;
            downloadBtn.classList.remove('hidden');
            if (saveModelBtn) saveModelBtn.classList.remove('hidden');
            
            // Complete immediately (no polling needed for generate-stl)
            setTimeout(() => {
                hideProgress();
                resetApplyButton();
            }, 500);
        } else {
            showError('Failed to generate STL from drawing');
            resetApplyButton();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Failed to apply drawing: ' + error.message);
        resetApplyButton();
    });
}

function resetApplyButton() {
    applyDrawingBtn.disabled = false;
    applyDrawingBtn.textContent = 'Apply Drawing';
}

function resetGenerateButton() {
    generateBtn.disabled = false;
    generateBtn.textContent = 'Generate STL';
}

// Model Storage Functions
function loadSavedModels() {
    fetch(`${API_BASE}/api/models`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displaySavedModels(data.models);
            } else {
                console.error('Failed to load models:', data.error);
            }
        })
        .catch(error => {
            console.error('Error loading models:', error);
        });
}

function displaySavedModels(models) {
    savedModelsGrid.innerHTML = '';
    
    if (models.length === 0) {
        noModelsMessage.classList.remove('hidden');
        return;
    }
    
    noModelsMessage.classList.add('hidden');
    
    models.forEach(model => {
        const modelCard = document.createElement('div');
        modelCard.className = 'saved-model-card';
        modelCard.innerHTML = `
            <div class="saved-model-preview">
                <img src="${API_BASE}/api/models/${model.id}/preview" alt="${model.name}" 
                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'200\' height=\'200\'%3E%3Crect width=\'200\' height=\'200\' fill=\'%23ddd\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\'%3ENo Preview%3C/text%3E%3C/svg%3E'">
            </div>
            <div class="saved-model-info">
                <h3>${escapeHtml(model.name || 'Untitled Model')}</h3>
                ${model.description ? `<p class="saved-model-description">${escapeHtml(model.description)}</p>` : ''}
                <div class="saved-model-meta">
                    <span>Thickness: ${model.thickness}mm</span>
                    <span>${model.solid_infill ? 'Solid' : 'With Holes'}</span>
                </div>
                <div class="saved-model-actions">
                    <button class="btn-download-small" onclick="downloadSavedModel('${model.id}', '${escapeHtml(model.name || 'model')}')">
                        📥 Download STL
                    </button>
                    <button class="btn-delete-small" onclick="deleteSavedModel('${model.id}')">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
        savedModelsGrid.appendChild(modelCard);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function downloadSavedModel(modelId, modelName) {
    window.location.href = `${API_BASE}/api/models/${modelId}/stl`;
}

function deleteSavedModel(modelId) {
    if (!confirm('Are you sure you want to delete this model?')) {
        return;
    }
    
    fetch(`${API_BASE}/api/models/${modelId}`, {
        method: 'DELETE'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadSavedModels(); // Refresh the list
            } else {
                alert('Failed to delete model: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error deleting model:', error);
            alert('Error deleting model');
        });
}

function showSaveModelModal() {
    if (!currentStlPath || !currentPreviewImage) {
        alert('No model to save. Please generate a model first.');
        return;
    }
    
    // Pre-fill with current description if available
    const textInputValue = textInput.value.trim();
    if (textInputValue) {
        modelNameInput.value = textInputValue.substring(0, 50);
        modelDescriptionInput.value = textInputValue;
    }
    
    saveModelModal.classList.remove('hidden');
}

function hideSaveModelModal() {
    saveModelModal.classList.add('hidden');
    modelNameInput.value = '';
    modelDescriptionInput.value = '';
}

function saveCurrentModel() {
    const name = modelNameInput.value.trim() || 'Untitled Model';
    const description = modelDescriptionInput.value.trim();
    
    if (!currentStlPath || !currentPreviewImage) {
        alert('No model to save');
        return;
    }
    
    confirmSaveBtn.disabled = true;
    confirmSaveBtn.textContent = 'Saving...';
    
    fetch(`${API_BASE}/api/models/save`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: name,
            description: description,
            preview_image: currentPreviewImage,
            stl_path: currentStlPath,
            thickness: parseFloat(thicknessInput.value) || 10.0,
            solid_infill: getSolidInfill()
        })
    })
        .then(response => response.json())
        .then(data => {
            confirmSaveBtn.disabled = false;
            confirmSaveBtn.textContent = 'Save';
            
            if (data.success) {
                hideSaveModelModal();
                loadSavedModels(); // Refresh the list
                alert('Model saved successfully!');
            } else {
                alert('Failed to save model: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error saving model:', error);
            confirmSaveBtn.disabled = false;
            confirmSaveBtn.textContent = 'Save';
            alert('Error saving model');
        });
}

// Update download handler to store model data
const originalDownloadHandler = downloadBtn.onclick;
downloadBtn.addEventListener('click', function() {
    // Store current model data when downloading
    if (downloadBtn.dataset.stlPath) {
        currentStlPath = downloadBtn.dataset.stlPath;
        if (currentImageBase64) {
            currentPreviewImage = currentImageBase64;
        }
    }
    handleDownload();
});

// Event listeners for model storage
if (saveModelBtn) {
    saveModelBtn.addEventListener('click', showSaveModelModal);
}
if (refreshModelsBtn) {
    refreshModelsBtn.addEventListener('click', loadSavedModels);
}
if (confirmSaveBtn) {
    confirmSaveBtn.addEventListener('click', saveCurrentModel);
}
if (cancelSaveBtn) {
    cancelSaveBtn.addEventListener('click', hideSaveModelModal);
}

// Helper function to handle STL response (single or multi-layer)
function handleStlResponse(data) {
    // Show G-code controls when STL is ready
    showGcodeControls();
    if (data.multi_layer && data.stl_paths) {
        // Multi-layer mode
        currentStlPaths = data.stl_paths; // Store paths globally
        currentLayerInfo = data.layer_info || []; // Store layer info globally
        displayLayers(data.stl_paths, currentLayerInfo);
        currentStlPath = data.stl_paths[0]; // Store first for save
        currentPreviewImage = data.preview_image;
        downloadBtn.classList.add('hidden');
        const downloadAllBtn = document.getElementById('downloadAllLayersBtn');
        if (downloadAllBtn) downloadAllBtn.classList.remove('hidden');
    } else {
        // Single STL mode
        currentStlPaths = null; // Clear multi-layer data
        currentLayerInfo = null;
        downloadBtn.dataset.stlPath = data.stl_path;
        currentStlPath = data.stl_path;
        currentPreviewImage = data.preview_image;
        downloadBtn.classList.remove('hidden');
        const downloadAllBtn = document.getElementById('downloadAllLayersBtn');
        if (downloadAllBtn) downloadAllBtn.classList.add('hidden');
        const layersList = document.getElementById('layersList');
        if (layersList) layersList.classList.add('hidden');
    }
    if (saveModelBtn) saveModelBtn.classList.remove('hidden');
    reverseBtn.classList.remove('hidden');
    updateReverseButtonText();
}

// Multi-layer STL functions
function displayLayers(stlPaths, layerInfo) {
    const layersList = document.getElementById('layersList');
    const layersContainer = document.getElementById('layersContainer');
    
    if (!layersList || !layersContainer) return;
    
    layersContainer.innerHTML = '';
    
    stlPaths.forEach((stlPath, index) => {
        const layerName = layerInfo[index]?.name || `Layer_${index + 1}`;
        const layerDiv = document.createElement('div');
        layerDiv.className = 'layer-item';
        // Escape quotes in layer name and path for onclick
        const safeLayerName = layerName.replace(/'/g, "\\'");
        const safeStlPath = stlPath.replace(/'/g, "\\'");
        layerDiv.innerHTML = `
            <span class="layer-name">${safeLayerName}</span>
            <button class="btn-download-layer" onclick="downloadLayer('${safeStlPath}', '${safeLayerName}')">
                📥 Download
            </button>
        `;
        layersContainer.appendChild(layerDiv);
    });
    
    layersList.classList.remove('hidden');
}

function downloadLayer(stlPath, layerName) {
    window.location.href = `${API_BASE}/api/download/${encodeURIComponent(stlPath)}`;
}

function downloadAllLayers() {
    // Use stored data instead of parsing HTML
    if (!currentStlPaths || currentStlPaths.length === 0) {
        alert('No STL files available. Please generate STL files first.');
        return;
    }
    
    // Prepare layer info
    const layerInfo = currentLayerInfo || currentStlPaths.map((path, index) => ({
        name: `Layer_${index + 1}`,
        index: index
    }));
    
    // Download as ZIP
    fetch(`${API_BASE}/api/download-layers`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            stl_paths: currentStlPaths,
            layer_info: layerInfo
        })
    })
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        return response.json().then(err => {
            throw new Error(err.error || 'Failed to download layers');
        });
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'layers.zip';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    })
    .catch(error => {
        console.error('Error downloading layers:', error);
        alert('Failed to download layers: ' + error.message);
    });
}

// Event listener for download all layers button
const downloadAllLayersBtn = document.getElementById('downloadAllLayersBtn');
if (downloadAllLayersBtn) {
    downloadAllLayersBtn.addEventListener('click', downloadAllLayers);
}

// G-code functionality
let currentGcodePath = null;
let currentGcodeSettings = null;

// G-code DOM elements
const generateGcodeBtn = document.getElementById('generateGcodeBtn');
const downloadGcodeBtn = document.getElementById('downloadGcodeBtn');
const gcodeSection = document.getElementById('gcodeSection');
const gcodeLearnFile = document.getElementById('gcodeLearnFile');
const learnGcodeBtn = document.getElementById('learnGcodeBtn');
const viewSettingsBtn = document.getElementById('viewSettingsBtn');
const gcodeSettingsModal = document.getElementById('gcodeSettingsModal');
const gcodeSettingsForm = document.getElementById('gcodeSettingsForm');
const saveGcodeSettingsBtn = document.getElementById('saveGcodeSettingsBtn');
const cancelGcodeSettingsBtn = document.getElementById('cancelGcodeSettingsBtn');

// Learn settings from G-code file
function learnGcodeSettings() {
    const file = gcodeLearnFile.files[0];
    if (!file) {
        alert('Please select a G-code file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('gcode_file', file);
    
    learnGcodeBtn.disabled = true;
    learnGcodeBtn.textContent = 'Learning...';
    
    fetch(`${API_BASE}/api/gcode/learn`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            alert('Settings learned successfully!');
            currentGcodeSettings = data.settings;
            gcodeLearnFile.value = ''; // Clear file input
        }
    })
    .catch(error => {
        console.error('Error learning G-code:', error);
        alert('Failed to learn settings: ' + error.message);
    })
    .finally(() => {
        learnGcodeBtn.disabled = false;
        learnGcodeBtn.textContent = 'Learn Settings';
    });
}

// Load current G-code settings
function loadGcodeSettings() {
    fetch(`${API_BASE}/api/gcode/settings`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentGcodeSettings = data.settings;
        }
    })
    .catch(error => {
        console.error('Error loading G-code settings:', error);
    });
}

// View/edit G-code settings
function viewGcodeSettings() {
    if (!currentGcodeSettings) {
        loadGcodeSettings().then(() => {
            if (currentGcodeSettings) {
                displayGcodeSettingsModal();
            }
        });
    } else {
        displayGcodeSettingsModal();
    }
}

// Unit mapping for G-code parameters
const gcodeUnits = {
    'temperatures': {
        'bed_temperature': '°C',
        'first_layer_bed_temperature': '°C',
        'extruder_temperature': '°C',
        'first_layer_temperature': '°C'
    },
    'layer': {
        'height': 'mm',
        'first_height': 'mm'
    },
    'extrusion': {
        'external_perimeter': 'mm',
        'perimeter': 'mm',
        'infill': 'mm',
        'solid_infill': 'mm',
        'top_infill': 'mm',
        'first_layer': 'mm'
    },
    'speeds': {
        'perimeter': 'mm/s',
        'external_perimeter': 'mm/s',
        'infill': 'mm/s',
        'solid_infill': 'mm/s',
        'top_solid_infill': 'mm/s',
        'support_material': 'mm/s',
        'travel': 'mm/s',
        'first_layer': 'mm/s'
    },
    'retraction': {
        'length': 'mm',
        'lift': 'mm',
        'speed': 'mm/s',
        'deretract_speed': 'mm/s'
    },
    'infill': {
        'pattern': '', // No unit for pattern (string)
        'density': '%'
    },
    'printer': {
        'nozzle_diameter': 'mm',
        'bed_shape': '' // No unit for bed_shape (string)
    }
};

function displayGcodeSettingsModal() {
    gcodeSettingsForm.innerHTML = '';
    
    // Create form fields for settings
    const categories = ['temperatures', 'layer', 'extrusion', 'speeds', 'retraction', 'infill', 'printer'];
    
    categories.forEach(category => {
        if (currentGcodeSettings[category]) {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'settings-category';
            categoryDiv.innerHTML = `<h4>${category.charAt(0).toUpperCase() + category.slice(1)}</h4>`;
            
            Object.keys(currentGcodeSettings[category]).forEach(key => {
                const value = currentGcodeSettings[category][key];
                const inputGroup = document.createElement('div');
                inputGroup.className = 'form-group';
                
                // Get unit for this parameter
                const unit = (gcodeUnits[category] && gcodeUnits[category][key]) || '';
                const unitDisplay = unit ? ` (${unit})` : '';
                
                const label = document.createElement('label');
                label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) + unitDisplay;
                label.setAttribute('for', `gcode_${category}_${key}`);
                
                let input;
                // Check if this is infill pattern - use dropdown
                if (category === 'infill' && key === 'pattern') {
                    input = document.createElement('select');
                    input.id = `gcode_${category}_${key}`;
                    input.className = 'gcode-setting-input';
                    input.dataset.category = category;
                    input.dataset.key = key;
                    
                    // Add infill pattern options
                    const patterns = [
                        { value: 'rectilinear', label: 'Rectilinear' },
                        { value: 'grid', label: 'Grid' },
                        { value: 'triangles', label: 'Triangles' },
                        { value: 'honeycomb', label: 'Honeycomb' }
                    ];
                    
                    patterns.forEach(pattern => {
                        const option = document.createElement('option');
                        option.value = pattern.value;
                        option.textContent = pattern.label;
                        if (pattern.value === value) {
                            option.selected = true;
                        }
                        input.appendChild(option);
                    });
                } else if (typeof value === 'string' && (key === 'bed_shape')) {
                    // Text input for bed_shape
                    input = document.createElement('input');
                    input.type = 'text';
                    input.id = `gcode_${category}_${key}`;
                    input.value = value;
                    input.className = 'gcode-setting-input';
                    input.dataset.category = category;
                    input.dataset.key = key;
                } else {
                    // Number input for other values
                    input = document.createElement('input');
                    input.type = 'number';
                    input.step = category === 'layer' || category === 'extrusion' || category === 'retraction' || (category === 'printer' && key === 'nozzle_diameter') ? '0.01' : '1';
                    input.id = `gcode_${category}_${key}`;
                    input.value = value;
                    input.className = 'gcode-setting-input';
                    input.dataset.category = category;
                    input.dataset.key = key;
                    
                    // Add unit as placeholder or suffix if it's a number input
                    if (unit) {
                        input.placeholder = unit;
                    }
                }
                
                inputGroup.appendChild(label);
                inputGroup.appendChild(input);
                categoryDiv.appendChild(inputGroup);
            });
            
            gcodeSettingsForm.appendChild(categoryDiv);
        }
    });
    
    gcodeSettingsModal.classList.remove('hidden');
}

function saveGcodeSettings() {
    const settings = {};
    
    // Collect all input values
    document.querySelectorAll('.gcode-setting-input').forEach(input => {
        const category = input.dataset.category;
        const key = input.dataset.key;
        // Handle select dropdowns (for infill pattern) vs number/text inputs
        let value;
        if (input.tagName === 'SELECT') {
            value = input.value; // String value for dropdowns
        } else if (input.type === 'number') {
            value = parseFloat(input.value) || 0;
        } else {
            value = input.value; // String value for text inputs
        }
        
        if (!settings[category]) {
            settings[category] = {};
        }
        settings[category][key] = value;
    });
    
    fetch(`${API_BASE}/api/gcode/settings`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ settings: settings })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            currentGcodeSettings = data.settings;
            gcodeSettingsModal.classList.add('hidden');
            alert('Settings saved successfully!');
        }
    })
    .catch(error => {
        console.error('Error saving G-code settings:', error);
        alert('Failed to save settings: ' + error.message);
    });
}

// Generate G-code from STL
function generateGcode() {
    if (!currentStlPath && !currentStlPaths) {
        alert('No STL file available. Please generate an STL first.');
        return;
    }
    
    // Use first STL path if multiple layers, otherwise use single path
    const stlPath = currentStlPaths && currentStlPaths.length > 0 ? currentStlPaths[0] : currentStlPath;
    
    if (!stlPath) {
        alert('STL file path not found');
        return;
    }
    
    generateGcodeBtn.disabled = true;
    generateGcodeBtn.textContent = 'Generating...';
    
    fetch(`${API_BASE}/api/gcode/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            stl_path: stlPath,
            job_id: currentJobId || Date.now().toString(),
            settings: currentGcodeSettings
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            currentGcodePath = data.gcode_filename;
            downloadGcodeBtn.classList.remove('hidden');
            alert('G-code generated successfully!');
        }
    })
    .catch(error => {
        console.error('Error generating G-code:', error);
        alert('Failed to generate G-code: ' + error.message);
    })
    .finally(() => {
        generateGcodeBtn.disabled = false;
        generateGcodeBtn.textContent = '🖨️ Generate G-code';
    });
}

// Download G-code file
function downloadGcode() {
    if (!currentGcodePath) {
        alert('No G-code file available');
        return;
    }
    
    window.location.href = `${API_BASE}/api/gcode/download/${currentGcodePath}`;
}

// Event listeners for G-code
if (learnGcodeBtn) {
    learnGcodeBtn.addEventListener('click', learnGcodeSettings);
}

if (viewSettingsBtn) {
    viewSettingsBtn.addEventListener('click', viewGcodeSettings);
}

if (saveGcodeSettingsBtn) {
    saveGcodeSettingsBtn.addEventListener('click', saveGcodeSettings);
}

if (cancelGcodeSettingsBtn) {
    cancelGcodeSettingsBtn.addEventListener('click', () => {
        gcodeSettingsModal.classList.add('hidden');
    });
}

if (generateGcodeBtn) {
    generateGcodeBtn.addEventListener('click', generateGcode);
}

if (downloadGcodeBtn) {
    downloadGcodeBtn.addEventListener('click', downloadGcode);
}

// Show G-code section and generate button when STL is ready
function showGcodeControls() {
    if (gcodeSection) {
        gcodeSection.classList.remove('hidden');
    }
    if (generateGcodeBtn) {
        generateGcodeBtn.classList.remove('hidden');
    }
}

// Load G-code settings on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        loadSavedModels();
        loadGcodeSettings();
    });
} else {
    loadSavedModels();
    loadGcodeSettings();
}
