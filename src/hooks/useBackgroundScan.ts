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
  scanInterval?: number; // How often to check if cache needs refresh (default: 3000ms)
  cacheValidityMs?: number; // How long cached results are valid (default: 15000ms = 15s)
  refreshBeforeExpiryMs?: number; // Refresh cache this many ms before expiry (default: 3000ms)
}

export function useBackgroundScan({
  videoRef,
  enabled,
  scanInterval = 3000, // Check every 3 seconds
  cacheValidityMs = 15000, // Cache valid for 15 seconds
  refreshBeforeExpiryMs = 3000, // Refresh 3 seconds before expiry
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

  // Check if cache needs refreshing
  const shouldRefreshCache = useCallback((): boolean => {
    const cached = cachedResultRef.current;
    if (!cached) {
      return true; // No cache, need to scan
    }

    const age = Date.now() - cached.timestamp;
    const timeUntilExpiry = cacheValidityMs - age;
    
    // Refresh if cache is expired or about to expire
    if (age >= cacheValidityMs) {
      console.log(`⏰ Cache expired (${(age / 1000).toFixed(1)}s old), refreshing...`);
      cachedResultRef.current = null;
      return true;
    }

    // Refresh if cache is close to expiry
    if (timeUntilExpiry <= refreshBeforeExpiryMs) {
      console.log(`🔄 Cache expiring soon (${(timeUntilExpiry / 1000).toFixed(1)}s remaining), refreshing...`);
      return true;
    }

    // Cache is still fresh, no need to refresh
    return false;
  }, [cacheValidityMs, refreshBeforeExpiryMs]);

  const performBackgroundScan = useCallback(async () => {
    if (isScanningRef.current) {
      console.log('⏸️ Background scan skipped - previous scan still in progress');
      return;
    }

    // Check if we need to refresh cache
    if (!shouldRefreshCache()) {
      const cached = cachedResultRef.current;
      const age = cached ? Date.now() - cached.timestamp : 0;
      console.log(`💾 Cache still valid (${(age / 1000).toFixed(1)}s old, ${((cacheValidityMs - age) / 1000).toFixed(1)}s remaining), skipping API call`);
      return;
    }

    // Double-check video is ready and actually playing
    if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      console.log('⏸️ Background scan skipped - video not ready');
      return;
    }

    // Additional check: ensure video is actually playing (not just metadata loaded)
    if (videoRef.current.readyState < 2) { // HAVE_CURRENT_DATA or higher
      console.log('⏸️ Background scan skipped - video not playing yet');
      return;
    }

    // Capture frames first BEFORE marking as scanning
    // This prevents race conditions where isScanningRef is set but scan fails
    const frames: string[] = [];
    const frameCount = 5; // Use fewer frames for background scanning
    const captureInterval = 200; // Capture every 200ms

    // Try to capture frames with retries
    for (let i = 0; i < frameCount; i++) {
      let frame = captureFrame();
      // Retry once if frame capture fails
      if (!frame && i === 0) {
        console.log('🔄 Retrying frame capture after brief delay...');
        await new Promise(resolve => setTimeout(resolve, 300));
        frame = captureFrame();
      }
      
      if (frame) {
        frames.push(frame);
      } else {
        console.warn(`⚠️ Failed to capture frame ${i + 1} for background scan`);
      }
      if (i < frameCount - 1) {
        await new Promise(resolve => setTimeout(resolve, captureInterval));
      }
    }

    // Only proceed if we have enough frames
    if (frames.length < 3) {
      console.warn(`⚠️ Only captured ${frames.length} frames, skipping background scan (need at least 3)`);
      // Don't set isScanningRef.current since we're not actually scanning
      return;
    }

    // Now mark as scanning since we have enough frames to proceed
    isScanningRef.current = true;
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      console.log(`🔄 Background scan: analyzing ${frames.length} frames...`);
      const startTime = performance.now();

      // Use the analyzeBatch function which has better error handling
      const data = await analyzeBatch(frames, 3);
      const scanTime = performance.now() - startTime;

      // Always save results if successful, regardless of abort state
      // (We want to cache results even if user started manual scan)
      if (data.success && data.results) {
        cachedResultRef.current = {
          response: data,
          timestamp: Date.now(),
          frames: frames,
        };
        console.log(`✅ Background scan complete in ${(scanTime / 1000).toFixed(2)}s - results cached for ${(cacheValidityMs / 1000).toFixed(1)}s`);
        
        // Check if this scan was superseded (only for logging, but still save cache)
        if (abortControllerRef.current !== abortController) {
          console.log('💾 Background scan completed but was superseded - cache still saved');
        }
      } else {
        console.warn('⚠️ Background scan returned unsuccessful response:', data.message);
        // Keep old cache if new scan failed
        if (cachedResultRef.current) {
          console.log('💾 Keeping previous cached result');
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError' || error.message?.includes('aborted')) {
        console.log('⏹️ Background scan aborted');
      } else {
        console.error('❌ Background scan error:', error.message || error);
      }
      // Keep old cache if new scan failed
      if (cachedResultRef.current) {
        console.log('💾 Keeping previous cached result due to error');
      }
    } finally {
      // Only reset if this is still the current scan
      if (abortControllerRef.current === abortController) {
        isScanningRef.current = false;
        abortControllerRef.current = null;
      }
    }
  }, [captureFrame, shouldRefreshCache]);

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

  // Wait for background scan to complete if one is in progress
  const waitForBackgroundScan = useCallback(async (timeoutMs: number = 3000): Promise<boolean> => {
    if (!isScanningRef.current) {
      return false; // No scan in progress
    }

    console.log('⏳ Waiting for background scan to complete...');
    const startTime = Date.now();
    
    // Poll until scan completes or timeout
    while (isScanningRef.current && (Date.now() - startTime) < timeoutMs) {
      await new Promise(resolve => setTimeout(resolve, 100)); // Check every 100ms
    }

    const completed = !isScanningRef.current;
    if (completed) {
      console.log('✅ Background scan completed while waiting');
    } else {
      console.log(`⏱️ Timeout waiting for background scan (${timeoutMs}ms)`);
    }
    
    return completed;
  }, []);

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
    waitForBackgroundScan,
  };
}

