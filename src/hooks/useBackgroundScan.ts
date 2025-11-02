import { useRef, useEffect, useCallback } from 'react';
import { analyzeBatch, AnalyzeResponse } from '@/lib/api';

interface CachedResult {
  response: AnalyzeResponse;
  timestamp: number;
  frames: string[];
}

interface UseBackgroundScanOptions {
  videoRef: React.RefObject<HTMLVideoElement>;
  enabled: boolean;
  scanInterval?: number; // How often to scan in ms (default: 2500ms)
  cacheValidityMs?: number; // How long cached results are valid (default: 5000ms)
}

export function useBackgroundScan({
  videoRef,
  enabled,
  scanInterval = 2500,
  cacheValidityMs = 5000,
}: UseBackgroundScanOptions) {
  const cachedResultRef = useRef<CachedResult | null>(null);
  const isScanningRef = useRef(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const captureFrame = useCallback((): string | null => {
    if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      return null;
    }

    try {
      const canvas = document.createElement('canvas');
      const maxWidth = 640;
      const maxHeight = 480;

      const videoWidth = videoRef.current.videoWidth;
      const videoHeight = videoRef.current.videoHeight;
      const aspectRatio = videoWidth / videoHeight;

      let canvasWidth = maxWidth;
      let canvasHeight = maxWidth / aspectRatio;

      if (canvasHeight > maxHeight) {
        canvasHeight = maxHeight;
        canvasWidth = maxHeight * aspectRatio;
      }

      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) return null;

      ctx.drawImage(videoRef.current, 0, 0, canvasWidth, canvasHeight);
      return canvas.toDataURL('image/jpeg', 0.3);
    } catch (error) {
      console.error('Error capturing frame:', error);
      return null;
    }
  }, [videoRef]);

  const performBackgroundScan = useCallback(async () => {
    if (isScanningRef.current) {
      console.log('⏸️ Background scan skipped - previous scan still in progress');
      return;
    }

    // Double-check video is ready
    if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      console.log('⏸️ Background scan skipped - video not ready');
      return;
    }

    // Capture a small batch of frames quickly
    const frames: string[] = [];
    const frameCount = 5; // Use fewer frames for background scanning
    const captureInterval = 200; // Capture every 200ms

    for (let i = 0; i < frameCount; i++) {
      const frame = captureFrame();
      if (frame) {
        frames.push(frame);
      } else {
        console.warn(`⚠️ Failed to capture frame ${i + 1} for background scan`);
      }
      if (i < frameCount - 1) {
        await new Promise(resolve => setTimeout(resolve, captureInterval));
      }
    }

    if (frames.length === 0) {
      console.warn('⚠️ No frames captured for background scan');
      return;
    }

    if (frames.length < 3) {
      console.warn(`⚠️ Only captured ${frames.length} frames, skipping background scan (need at least 3)`);
      return;
    }

    isScanningRef.current = true;
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      console.log(`🔄 Background scan: analyzing ${frames.length} frames...`);
      const startTime = performance.now();

      // Use the analyzeBatch function which has better error handling
      const data = await analyzeBatch(frames, 3);
      const scanTime = performance.now() - startTime;

      // Check if request was aborted
      if (abortControllerRef.current !== abortController) {
        console.log('⏹️ Background scan aborted (new scan started)');
        return;
      }

      if (data.success && data.results) {
        cachedResultRef.current = {
          response: data,
          timestamp: Date.now(),
          frames: frames,
        };
        console.log(`✅ Background scan complete in ${(scanTime / 1000).toFixed(2)}s - results cached`);
      } else {
        console.warn('⚠️ Background scan returned unsuccessful response:', data.message);
      }
    } catch (error: any) {
      if (error.name === 'AbortError' || error.message?.includes('aborted')) {
        console.log('⏹️ Background scan aborted');
      } else {
        console.error('❌ Background scan error:', error.message || error);
      }
    } finally {
      // Only reset if this is still the current scan
      if (abortControllerRef.current === abortController) {
        isScanningRef.current = false;
        abortControllerRef.current = null;
      }
    }
  }, [captureFrame]);

  // Get cached result if available and fresh
  const getCachedResult = useCallback((): CachedResult | null => {
    const cached = cachedResultRef.current;
    if (!cached) return null;

    const age = Date.now() - cached.timestamp;
    if (age > cacheValidityMs) {
      console.log(`⏰ Cached result expired (${(age / 1000).toFixed(1)}s old, max ${(cacheValidityMs / 1000).toFixed(1)}s)`);
      cachedResultRef.current = null;
      return null;
    }

    console.log(`✅ Using cached result (${(age / 1000).toFixed(1)}s old)`);
    return cached;
  }, [cacheValidityMs]);

  // Clear cache
  const clearCache = useCallback(() => {
    cachedResultRef.current = null;
    console.log('🗑️ Cache cleared');
  }, []);

  // Stop background scanning
  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    isScanningRef.current = false;
  }, []);

  // Start background scanning
  useEffect(() => {
    if (!enabled) {
      stop();
      return;
    }

    // Verify video is actually ready before starting
    const checkAndStart = () => {
      if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
        console.log('⏸️ Background scan: video not ready, retrying...');
        return false;
      }
      
      console.log(`🚀 Starting background scanning (video: ${videoRef.current.videoWidth}x${videoRef.current.videoHeight})...`);
      
      // Perform initial scan immediately
      performBackgroundScan();

      // Then scan periodically
      intervalRef.current = setInterval(() => {
        performBackgroundScan();
      }, scanInterval);
      
      return true;
    };

    // Try to start immediately if video is ready
    if (!checkAndStart()) {
      // If not ready, wait a bit and try again
      const startDelay = setTimeout(() => {
        if (!checkAndStart()) {
          // Try one more time after another delay
          const retryDelay = setTimeout(() => {
            checkAndStart();
          }, 1000);
          return () => clearTimeout(retryDelay);
        }
      }, 1500);

      return () => {
        clearTimeout(startDelay);
        stop();
      };
    }

    return () => {
      stop();
    };
  }, [enabled, scanInterval, performBackgroundScan, stop, videoRef]);

  return {
    getCachedResult,
    clearCache,
    isScanning: () => isScanningRef.current,
  };
}

