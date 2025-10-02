import { useCallback, useState, useEffect, useRef } from "react";
import { Upload, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";

interface UploadZoneProps {
  onImageSelect: (file: File) => void;
}

const UploadZone = ({ onImageSelect }: UploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
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
    };
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) {
        onImageSelect(file);
      }
    },
    [onImageSelect]
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
        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  return (
    <div className="relative min-h-screen flex items-center justify-center p-8 overflow-hidden">
      {/* Camera feed background */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Dark overlay for better text contrast */}
      <div className="absolute inset-0 bg-black/30" />

      {/* Content */}
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
                Upload your photo
              </h2>
              <p className="text-white/80 drop-shadow">
                Drag and drop your image here, or click to browse
              </p>
            </div>

            <div className="flex gap-4">
              <Button
                size="lg"
                className="relative overflow-hidden bg-white/90 hover:bg-white text-primary backdrop-blur-sm shadow-lg"
                onClick={() => document.getElementById("file-input")?.click()}
              >
                <Upload className="w-5 h-5 mr-2" />
                Choose File
              </Button>

              <Button
                size="lg"
                className="bg-white/20 hover:bg-white/30 text-white border-2 border-white/40 backdrop-blur-sm shadow-lg"
                onClick={() => document.getElementById("camera-input")?.click()}
              >
                <Camera className="w-5 h-5 mr-2" />
                Take Photo
              </Button>
            </div>

            <input
              id="file-input"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileInput}
            />
            <input
              id="camera-input"
              type="file"
              accept="image/*"
              capture="user"
              className="hidden"
              onChange={handleFileInput}
            />
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
    </div>
  );
};

export default UploadZone;
