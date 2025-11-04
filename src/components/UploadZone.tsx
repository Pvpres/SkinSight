import { useCallback, useState, useEffect, useRef } from "react";
import { Upload, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useBackgroundScan } from "@/hooks/useBackgroundScan";

interface UploadZoneProps {
  onUploadFile: (file: File) => void;
  onScanFrames: (frames: string[]) => void;
  onStartScan: () => void;
  onUseCachedResult?: (frames: string[], response: any) => void;
  readonly?: boolean;
  onNoFaceDetected?: () => void;
}

const UploadZone = ({ onUploadFile, onScanFrames, onStartScan, onUseCachedResult, readonly = false, onNoFaceDetected }: UploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [showRetryButton, setShowRetryButton] = useState(false);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          
          // Multiple checks to ensure video is ready
          const checkVideoReady = () => {
            if (videoRef.current && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
              setIsVideoReady(true);
              console.log(`📹 Video ready: ${videoRef.current.videoWidth}x${videoRef.current.videoHeight}`);
              return true;
            }
            return false;
          };

          // Check immediately
          if (!checkVideoReady()) {
            // Also listen for metadata loaded
            videoRef.current.onloadedmetadata = () => {
              checkVideoReady();
            };
            
            // Also check on playing event
            videoRef.current.onplaying = () => {
              checkVideoReady();
            };
            
            // Fallback: check after a short delay
            setTimeout(() => {
              checkVideoReady();
            }, 500);
          }
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
        setCameraError("Camera access denied");
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
      setIsVideoReady(false);
    };
  }, []);

  // Enable background scanning when camera is ready and not in readonly mode
  const backgroundScan = useBackgroundScan({
    videoRef,
    enabled: !readonly && !cameraError && isVideoReady,
  });

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onUploadFile(file);
      }
    },
    [onUploadFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        onUploadFile(file);
      }
    },
    [onUploadFile]
  );

  // Helper function to capture a single test frame for face detection
  const captureTestFrame = useCallback((): string | null => {
    if (!videoRef.current) return null;
    
    // Check if video is actually ready (has dimensions and is playing)
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) {
      return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    
    if (ctx) {
      ctx.drawImage(video, 0, 0);
      return canvas.toDataURL("image/jpeg", 0.7);
    }
    return null;
  }, []);

  // Helper function to wait for video to be ready
  const waitForVideoReady = useCallback(async (maxWaitMs: number = 2000): Promise<boolean> => {
    if (!videoRef.current) return false;
    
    const startTime = Date.now();
    while (Date.now() - startTime < maxWaitMs) {
      const video = videoRef.current;
      if (video && video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) {
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return false;
  }, []);

  const startFrameCapture = useCallback(async () => {
    if (!videoRef.current || isScanning) {
      console.log('⏸️ startFrameCapture skipped:', { hasVideo: !!videoRef.current, isScanning });
      return;
    }
    if (!videoRef.current.srcObject) {
      setCameraError("Camera not enabled");
      return;
    }

    // Hide retry button when starting new scan attempt
    setShowRetryButton(false);

    // Wait for video to be ready before attempting face detection
    console.log('⏳ Waiting for video to be ready...');
    const videoReady = await waitForVideoReady(2000);
    
    if (!videoReady) {
      console.warn('⚠️ Video not ready after waiting - showing retry button');
      setShowRetryButton(true);
      if (onNoFaceDetected) onNoFaceDetected();
      return;
    }

    // CRITICAL: Check for face detection BEFORE starting scan animation or API calls
    console.log('🔍 Pre-checking for face detection...');
    const testFrame = captureTestFrame();
    
    if (!testFrame) {
      console.warn('⚠️ Could not capture test frame (video may not be fully ready)');
      setShowRetryButton(true);
      if (onNoFaceDetected) onNoFaceDetected();
      return;
    }

    try {
      const { checkFramesForFace } = await import('@/lib/faceDetection');
      const hasFace = await checkFramesForFace([testFrame]);
      
      if (!hasFace) {
        console.log('🚫 No face detected - showing retry button, NOT starting scan');
        setShowRetryButton(true);
        if (onNoFaceDetected) onNoFaceDetected();
        // CRITICAL: Don't start scan animation or API calls - just show retry button
        return;
      }
      
      console.log('✅ Face detected - proceeding with scan');
    } catch (error) {
      console.error('❌ Face detection check failed:', error);
      // On error, allow scan to proceed (fallback)
    }

    // Face detected - proceed with normal flow
    console.log('🔍 Checking for cached result...');
    let cached = backgroundScan.getCachedResult();
    
    // If we have cached result, use it
    if (cached && onUseCachedResult) {
      console.log('⚡ Using cached results - showing animation...', {
        hasFrames: cached.frames?.length > 0,
        hasResponse: !!cached.response,
        responseSuccess: cached.response?.success
      });
      onStartScan();
      setIsScanning(true);
      try {
        onUseCachedResult(cached.frames, cached.response);
        console.log('✅ onUseCachedResult called successfully');
      } catch (error) {
        console.error('❌ Error calling onUseCachedResult:', error);
      }
      return;
    }

    // No cached result - ALWAYS proceed with frame capture
    // This ensures scan always happens when user presses scan button
    
    // CRITICAL: Wait for video to be ready AGAIN before starting frame capture
    // This ensures video is stable even if state changed after face detection
    console.log('⏳ Verifying video is ready before frame capture...');
    const videoReadyForCapture = await waitForVideoReady(1000);
    
    if (!videoReadyForCapture || !videoRef.current) {
      console.error('❌ Video not ready for frame capture - showing retry button');
      setShowRetryButton(true);
      if (onNoFaceDetected) onNoFaceDetected();
      return;
    }
    
    // Store stable reference to video element to avoid ref changes during capture
    const videoElement = videoRef.current;
    if (!videoElement.videoWidth || !videoElement.videoHeight || videoElement.readyState < 2) {
      console.error('❌ Video element not ready - showing retry button');
      setShowRetryButton(true);
      if (onNoFaceDetected) onNoFaceDetected();
      return;
    }
    
    console.log('🎬 No cache available - starting frame capture with stable video reference...');
    onStartScan();
    setIsScanning(true);
    
    const frames: string[] = [];
    const scanDuration = 3000; // 3s
    const targetFrameCount = 10;
    const frameCaptureInterval = Math.max(100, scanDuration / targetFrameCount);
    const startTime = Date.now();
    let consecutiveFailures = 0;
    const maxConsecutiveFailures = 3;
    
    // Safety timeout to ensure scan completes even if something goes wrong
    const safetyTimeout = setTimeout(() => {
      if (frames.length > 0) {
        console.log(`⏱️ Safety timeout reached with ${frames.length} frames - completing scan`);
        setIsScanning(false);
        onScanFrames(frames);
      } else {
        console.error('❌ Safety timeout reached with no frames - showing retry');
        setIsScanning(false);
        setShowRetryButton(true);
        if (onNoFaceDetected) onNoFaceDetected();
      }
    }, scanDuration + 2000); // 5 seconds total (3s scan + 2s buffer)
    
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      
      // Use stored video element reference, but also check current ref as fallback
      const video = videoRef.current || videoElement;
      
      if (!video || !video.videoWidth || !video.videoHeight || video.readyState < 2) {
        consecutiveFailures++;
        console.warn(`⚠️ Video not ready (failure ${consecutiveFailures}/${maxConsecutiveFailures})`);
        
        if (consecutiveFailures >= maxConsecutiveFailures) {
          console.error('❌ Too many consecutive failures - aborting frame capture');
          clearInterval(interval);
          clearTimeout(safetyTimeout);
          setIsScanning(false);
          setShowRetryButton(true);
          if (onNoFaceDetected) onNoFaceDetected();
          return;
        }
        return; // Skip this frame, try again next interval
      }
      
      // Reset failure counter on successful check
      consecutiveFailures = 0;
      
      const canvas = document.createElement("canvas");
      const maxWidth = 640;
      const maxHeight = 480;
      
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;
      const aspectRatio = videoWidth / videoHeight;
      
      let canvasWidth = maxWidth;
      let canvasHeight = maxWidth / aspectRatio;
      
      if (canvasHeight > maxHeight) {
        canvasHeight = maxHeight;
        canvasWidth = maxHeight * aspectRatio;
      }
      
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;
      
      const ctx = canvas.getContext("2d");
      if (ctx && video) {
        try {
          ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);
          frames.push(canvas.toDataURL("image/jpeg", 0.3));
          console.log(`📸 Captured frame ${frames.length} (${canvasWidth}x${canvasHeight})`);
        } catch (error) {
          console.error('❌ Error capturing frame:', error);
          consecutiveFailures++;
        }
      }
      
      if (elapsed >= scanDuration || frames.length >= targetFrameCount) {
        clearInterval(interval);
        clearTimeout(safetyTimeout);
        setIsScanning(false);
        console.log(`🎬 Frame capture complete: ${frames.length} frames`);
        
        if (frames.length > 0) {
          onScanFrames(frames);
        } else {
          console.error('❌ No frames captured - cannot proceed with scan');
          setShowRetryButton(true);
          if (onNoFaceDetected) onNoFaceDetected();
        }
      }
    }, frameCaptureInterval);
  }, [isScanning, onScanFrames, onStartScan, backgroundScan, onUseCachedResult, captureTestFrame, onNoFaceDetected, waitForVideoReady]);

  return (
    <div className="relative min-h-screen flex items-center justify-center p-8 overflow-hidden">
      {/* Camera feed background */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
      />

      {/* Dark overlay for better text contrast */}
      <div className="absolute inset-0 bg-black/30" />

      {/* Content */}
      {!readonly && (
      <div className="relative z-10 w-full max-w-2xl">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white drop-shadow-lg mb-4">
            SkinSight
          </h1>
          <p className="text-xl text-white/90 drop-shadow-md">
            AI-powered skin analysis for better care
          </p>
        </div>

        <div
          className={`
            relative border-2 border-dashed rounded-3xl p-16 transition-all duration-300 backdrop-blur-xl
            ${
              isDragging
                ? "border-success bg-white/20 scale-105"
                : "border-white/40 bg-white/10 hover:border-success/70 hover:bg-white/15"
            }
          `}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="flex flex-col items-center gap-6">
            <div className="w-24 h-24 rounded-full bg-white/20 backdrop-blur-md flex items-center justify-center">
              <Upload className="w-12 h-12 text-white" />
            </div>

            <div className="text-center">
              <h2 className="text-2xl font-semibold text-white drop-shadow-md mb-2">
                Scan or upload a photo
              </h2>
              <p className="text-white/80 drop-shadow">
                Use your camera for a quick 3s scan, or upload manually
              </p>
            </div>

            <div className="flex flex-col gap-4 items-center">
              {showRetryButton ? (
                <div className="flex flex-col items-center gap-4 w-full">
                  <div className="bg-amber-500/90 border-2 border-amber-400 rounded-lg p-4 text-center shadow-xl backdrop-blur-sm w-full max-w-md">
                    <p className="text-white text-sm font-semibold mb-2">
                      👤 No face detected
                    </p>
                    <p className="text-white/90 text-xs mb-4">
                      Please position your face clearly in the camera view and make sure it's centered and well-lit
                    </p>
                    <Button
                      size="lg"
                      className="bg-white/90 hover:bg-white text-amber-600 border-2 border-white backdrop-blur-sm shadow-lg font-semibold"
                      onClick={startFrameCapture}
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      Retry Scan
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex gap-4">
                  {!cameraError && (
                    <Button
                      size="lg"
                      disabled={isScanning || !isVideoReady}
                      className="bg-white/20 hover:bg-white/30 text-white border-2 border-white/40 backdrop-blur-sm shadow-lg"
                      onClick={startFrameCapture}
                    >
                      <Camera className="w-5 h-5 mr-2" />
                      {isScanning ? "Scanning..." : !isVideoReady ? "Loading camera..." : "Scan Face"}
                    </Button>
                  )}

                  <Button
                    size="lg"
                    className="relative overflow-hidden bg-white/90 hover:bg-white text-primary backdrop-blur-sm shadow-lg"
                    onClick={() => document.getElementById("file-input")?.click()}
                  >
                    <Upload className="w-5 h-5 mr-2" />
                    Upload Photo
                  </Button>
                </div>
              )}
            </div>

            <input
              id="file-input"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
            />
            {/* retain camera input for mobile capture fallback via file picker if desired */}
          </div>
        </div>

        <p className="text-center text-sm text-white/70 drop-shadow mt-6">
          We respect your privacy. Your images are analyzed securely and never stored.
        </p>

        {cameraError && (
          <p className="text-center text-sm text-white bg-destructive/80 backdrop-blur-sm px-4 py-2 rounded-lg mt-4">
            {cameraError} - You can still upload photos manually
          </p>
        )}
      </div>
      )}
    </div>
  );
};

export default UploadZone;
