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
}

const UploadZone = ({ onUploadFile, onScanFrames, onStartScan, onUseCachedResult, readonly = false }: UploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);

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

  const startFrameCapture = useCallback(() => {
    if (!videoRef.current || isScanning) return;
    if (!videoRef.current.srcObject) {
      setCameraError("Camera not enabled");
      return;
    }

    // Check for cached result first
    const cached = backgroundScan.getCachedResult();
    if (cached && onUseCachedResult) {
      console.log('⚡ Using cached results - showing animation...');
      onStartScan();
      setIsScanning(true);
      // Use cached results but let animation play for a reasonable duration
      onUseCachedResult(cached.frames, cached.response);
      return;
    }

    // No cached result, proceed with normal capture
    onStartScan();
    setIsScanning(true);
    const frames: string[] = [];
    const scanDuration = 3000; // 3s
    const targetFrameCount = 10; // Reduced from ~20 to ~10 frames
    const frameCaptureInterval = Math.max(100, scanDuration / targetFrameCount); // ~300ms for 10 frames
    const startTime = Date.now();
    console.log('🎬 Starting frame capture...');
    
    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      if (!videoRef.current) return;
      
      // Create smaller canvas for faster processing
      const canvas = document.createElement("canvas");
      const maxWidth = 640; // Limit size for faster processing
      const maxHeight = 480;
      
      // Calculate dimensions maintaining aspect ratio
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
      
      const ctx = canvas.getContext("2d");
      if (ctx) {
        // Draw with scaling for smaller image
        ctx.drawImage(videoRef.current, 0, 0, canvasWidth, canvasHeight);
        // Use lower quality for faster processing
        frames.push(canvas.toDataURL("image/jpeg", 0.3));
        console.log(`📸 Captured frame ${frames.length} (${canvasWidth}x${canvasHeight})`);
      }
      
      if (elapsed >= scanDuration || frames.length >= targetFrameCount) {
        clearInterval(interval);
        setIsScanning(false);
        console.log(`🎬 Frame capture complete: ${frames.length} frames`);
        onScanFrames(frames);
      }
    }, frameCaptureInterval);
  }, [isScanning, onScanFrames, onStartScan, backgroundScan, onUseCachedResult]);

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

            <div className="flex gap-4">
              {!cameraError && (
                <Button
                  size="lg"
                  disabled={isScanning}
                  className="bg-white/20 hover:bg-white/30 text-white border-2 border-white/40 backdrop-blur-sm shadow-lg"
                  onClick={startFrameCapture}
                >
                  <Camera className="w-5 h-5 mr-2" />
                  {isScanning ? "Scanning..." : "Scan Face"}
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
