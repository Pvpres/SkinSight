export const API_BASE = "https://pvpres-skinsightmodel.hf.space";

// Test API connectivity
export async function testApiConnection(): Promise<boolean> {
  try {
    console.log('🔍 Testing API connectivity...');
    const startTime = performance.now();
    
    const response = await fetch(`${API_BASE}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000) // 5s timeout for health check
    });
    
    const responseTime = performance.now() - startTime;
    console.log(`🏥 Health check completed in ${responseTime.toFixed(2)}ms`);
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ API is healthy:', data);
      return true;
    } else {
      console.warn('⚠️ API health check failed:', response.status);
      return false;
    }
  } catch (error) {
    console.error('❌ API connectivity test failed:', error);
    return false;
  }
}

export interface AnalyzeResponse<T = any> {
  success: boolean;
  message: string;
  results?: T;
  processing_time?: number;
}

export async function analyze(imageDataUrl: string, scanDurationSeconds: number = 3): Promise<AnalyzeResponse> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout
  
  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_data: imageDataUrl, scan_duration: scanDurationSeconds }),
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    }
    
    const data = await res.json();
    console.log('Analyze response:', data);
    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timeout - please try again');
    }
    throw error;
  }
}

export async function analyzeBatch(imageDataUrls: string[], scanDurationSeconds: number = 3): Promise<AnalyzeResponse> {
  const startTime = performance.now();
  console.log(`🚀 Starting batch analysis with ${imageDataUrls.length} frames at ${new Date().toISOString()}`);
  
  // Limit frames to prevent oversized payloads
  const maxFrames = 12;
  const framesToSend = imageDataUrls.slice(0, maxFrames);
  if (imageDataUrls.length > maxFrames) {
    console.warn(`⚠️ Limiting frames from ${imageDataUrls.length} to ${maxFrames} to reduce payload size`);
  }
  
  // Calculate approximate payload size
  const payloadSize = JSON.stringify({ image_data_list: framesToSend, scan_duration: scanDurationSeconds }).length;
  const payloadSizeMB = (payloadSize / (1024 * 1024)).toFixed(2);
  console.log(`📦 Payload size: ~${payloadSizeMB}MB (${framesToSend.length} frames)`);
  
  const controller = new AbortController();
  const timeoutMs = 60000; // 60s timeout for batch (increased from 45s)
  const timeoutId = setTimeout(() => {
    console.error(`⏱️ Request timeout after ${timeoutMs}ms - aborting`);
    controller.abort();
  }, timeoutMs);
  
  try {
    console.log('📤 Sending request to API...');
    const requestStart = performance.now();
    
    // Create the request body
    const requestBody = JSON.stringify({ 
      image_data_list: framesToSend, 
      scan_duration: scanDurationSeconds 
    });
    
    // Log request initiation
    console.log(`🌐 Initiating fetch request to ${API_BASE}/analyze-batch`);
    
    const res = await fetch(`${API_BASE}/analyze-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: requestBody,
      signal: controller.signal
    });
    
    const requestTime = performance.now() - requestStart;
    console.log(`📡 Request completed in ${requestTime.toFixed(2)}ms`);
    
    clearTimeout(timeoutId);
    
    if (!res.ok) {
      const errorText = await res.text().catch(() => res.statusText);
      throw new Error(`HTTP ${res.status}: ${errorText}`);
    }
    
    console.log('📥 Parsing response...');
    const parseStart = performance.now();
    const data = await res.json();
    const parseTime = performance.now() - parseStart;
    const totalTime = performance.now() - startTime;
    
    console.log(`✅ Batch analysis completed in ${totalTime.toFixed(2)}ms (request: ${requestTime.toFixed(2)}ms, parse: ${parseTime.toFixed(2)}ms)`);
    console.log('📊 Response data:', data);
    return data;
  } catch (error: any) {
    clearTimeout(timeoutId);
    const totalTime = performance.now() - startTime;
    console.error(`❌ Batch analysis failed after ${(totalTime / 1000).toFixed(2)}s:`, error);
    
    if (error.name === 'AbortError' || error.message?.includes('aborted')) {
      throw new Error(`Request timed out after ${(totalTime / 1000).toFixed(1)}s. The server may be processing too many frames. Try again or upload a single photo.`);
    }
    
    if (error.message?.includes('Failed to fetch') || error.message?.includes('NetworkError')) {
      throw new Error('Network error. Please check your connection and try again.');
    }
    
    throw error;
  }
}

export async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function compressImage(file: File, maxWidth: number = 800, quality: number = 0.8): Promise<string> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Calculate new dimensions
      const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
      canvas.width = img.width * ratio;
      canvas.height = img.height * ratio;
      
      // Draw and compress
      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
      const compressedDataUrl = canvas.toDataURL('image/jpeg', quality);
      resolve(compressedDataUrl);
    };
    
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}


