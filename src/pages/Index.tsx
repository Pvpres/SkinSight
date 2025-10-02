import { useState } from "react";
import UploadZone from "@/components/UploadZone";
import ScanAnimation from "@/components/ScanAnimation";
import FlashTransition from "@/components/FlashTransition";
import ResultsView from "@/components/ResultsView";

type AppState = "upload" | "scanning" | "flash" | "results";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("upload");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setAppState("scanning");
  };

  const handleScanComplete = () => {
    setAppState("flash");
  };

  const handleFlashComplete = () => {
    setAppState("results");
  };

  const handleNewScan = () => {
    setSelectedImage(null);
    setAppState("upload");
  };

  // Mock analysis results
  const analysisResults = {
    condition: "Healthy Skin",
    confidence: 94,
    description: "Your skin shows signs of good health with balanced hydration levels.",
  };

  return (
    <>
      {appState === "upload" && <UploadZone onImageSelect={handleImageSelect} />}
      
      <ScanAnimation
        isScanning={appState === "scanning"}
        onScanComplete={handleScanComplete}
      />
      
      <FlashTransition
        isActive={appState === "flash"}
        onComplete={handleFlashComplete}
      />
      
      {appState === "results" && (
        <div className="relative">
          <ResultsView
            condition={analysisResults.condition}
            confidence={analysisResults.confidence}
            description={analysisResults.description}
          />
          <button
            onClick={handleNewScan}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 px-6 py-3 bg-primary text-primary-foreground rounded-full font-medium hover:opacity-90 transition-opacity shadow-lg"
          >
            New Analysis
          </button>
        </div>
      )}
    </>
  );
};

export default Index;
