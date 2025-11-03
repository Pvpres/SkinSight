import { useCallback, useEffect, useState } from "react";
import UploadZone from "@/components/UploadZone";
import ScanAnimation from "@/components/ScanAnimation";
import FlashTransition from "@/components/FlashTransition";
import ResultsView from "@/components/ResultsView";
import { analyze, analyzeBatch, fileToDataUrl, compressImage, testApiConnection } from "@/lib/api";

type AppState = "upload" | "scanning" | "flash" | "results";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("upload");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<{
    condition: string;
    confidence: number;
    description: string;
  } | null>(null);
  const [apiDone, setApiDone] = useState(false);

  const handleUploadFile = async (file: File) => {
    setError(null);
    setResults(null);
    setApiDone(false);
    setSelectedImage(file);
    setAppState("scanning");
    
    try {
      console.log('Starting file analysis...');
      const dataUrl = await compressImage(file, 800, 0.8);
      console.log('File compressed and converted to data URL, calling API...');
      
      const response = await analyze(dataUrl, 3);
      console.log('API response received:', response);
      
      if (!response.success || !response.results) {
        setError(response.message || "Analysis failed");
        setApiDone(true);
        return;
      }
      
      const r: any = response.results;
      const isHealthy = r?.overall_assessment?.is_healthy ?? false;
      const condition = isHealthy ? "Healthy Skin" : (r?.prediction?.condition || "Detected Condition");
      const confidence = Math.round(
        r?.prediction?.confidence_percentage ?? r?.overall_assessment?.healthy_percentage ?? 0
      );
      const description = isHealthy
        ? "Your skin shows signs of good health with balanced hydration levels."
        : "Our analysis suggests a possible skin condition. Consider tailored care and, if needed, consult a professional.";
      
      console.log('Setting results:', { condition, confidence, description });
      setResults({ condition, confidence, description });
      setApiDone(true);
    } catch (e: any) {
      console.error('Upload file error:', e);
      setError(e?.message || "Network error");
      setApiDone(true);
    }
  };

  const processScanResults = useCallback((response: any) => {
    if (!response.success || !response.results) {
      setError(response.message || "Analysis failed");
      setApiDone(true);
      return;
    }
    
    const r: any = response.results;
    const isHealthy = r?.overall_assessment?.is_healthy ?? false;
    const condition = isHealthy ? "Healthy Skin" : (r?.prediction?.condition || "Detected Condition");
    const confidence = Math.round(
      r?.prediction?.confidence_percentage ?? r?.overall_assessment?.healthy_percentage ?? 0
    );
    const description = isHealthy
      ? "Your skin shows signs of good health with balanced hydration levels."
      : "Our analysis suggests a possible skin condition. Consider tailored care and, if needed, consult a professional.";
    
    console.log('Setting results:', { condition, confidence, description });
    setResults({ condition, confidence, description });
    setApiDone(true);
  }, []);

  const handleScanFrames = async (frames: string[]) => {
    try {
      console.log('Starting batch analysis with', frames.length, 'frames...');
      const response = await analyzeBatch(frames, 3);
      console.log('Batch API response received:', response);
      processScanResults(response);
    } catch (e: any) {
      console.error('Scan frames error:', e);
      setError(e?.message || "Network error");
      setApiDone(true);
    }
  };

  const handleUseCachedResult = useCallback((frames: string[], response: any) => {
    console.log('⚡ handleUseCachedResult called', {
      hasFrames: frames?.length > 0,
      hasResponse: !!response,
      responseSuccess: response?.success
    });
    
    // Process and store results immediately
    if (!response || !response.success || !response.results) {
      console.error('❌ Invalid cached response:', response);
      setError(response?.message || "Analysis failed");
      setApiDone(true);
      return;
    }
    
    const r: any = response.results;
    const isHealthy = r?.overall_assessment?.is_healthy ?? false;
    const condition = isHealthy ? "Healthy Skin" : (r?.prediction?.condition || "Detected Condition");
    const confidence = Math.round(
      r?.prediction?.confidence_percentage ?? r?.overall_assessment?.healthy_percentage ?? 0
    );
    const description = isHealthy
      ? "Your skin shows signs of good health with balanced hydration levels."
      : "Our analysis suggests a possible skin condition. Consider tailored care and, if needed, consult a professional.";
    
    console.log('💾 Storing results:', { condition, confidence, description });
    // Store results immediately
    setResults({ condition, confidence, description });
    
    // Set apiDone after a short delay to allow animation to start
    // The animation will progress naturally from 0% to 95%, then complete to 100% when apiDone becomes true
    // This gives a smooth visual experience while still completing quickly
    const animationDelay = 1500; // 1.5 seconds - enough for animation to progress smoothly
    console.log(`⏱️ Setting apiDone to true in ${animationDelay}ms`);
    
    // Use a ref or closure to ensure we're setting the right state
    const timeoutId = setTimeout(() => {
      console.log('✅ Results ready - setting apiDone to true');
      setApiDone(true);
      // Double-check it was set
      setTimeout(() => {
        console.log('🔍 Verifying apiDone was set correctly');
      }, 100);
    }, animationDelay);
    
    // Fallback: ensure apiDone is set even if something goes wrong
    setTimeout(() => {
      console.log('🛡️ Fallback: ensuring apiDone is true after 3 seconds');
      setApiDone(true);
    }, 3000);
  }, []);

  const handleStartScan = () => {
    setError(null);
    setResults(null);
    setApiDone(false);
    setAppState("scanning");
    console.log('Scan started - waiting for frames...');
  };

  const handleScanComplete = () => {
    setAppState("flash");
  };

  const handleFlashComplete = () => {
    setAppState("results");
  };

  const handleNewScan = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    setAppState("upload");
  };

  // Test API connectivity on component mount
  useEffect(() => {
    testApiConnection();
  }, []);

  // When ScanAnimation completes, proceed to flash, then results

  return (
    <>
      {appState === "upload" && (
        <UploadZone
          onUploadFile={handleUploadFile}
          onScanFrames={handleScanFrames}
          onStartScan={handleStartScan}
          onUseCachedResult={handleUseCachedResult}
        />
      )}
      {appState === "scanning" && (
        <UploadZone
          readonly
          onUploadFile={handleUploadFile}
          onScanFrames={handleScanFrames}
          onStartScan={handleStartScan}
          onUseCachedResult={handleUseCachedResult}
        />
      )}
      
      <ScanAnimation
        isScanning={appState === "scanning"}
        apiDone={apiDone}
        onScanComplete={handleScanComplete}
      />
      
      <FlashTransition
        isActive={appState === "flash"}
        onComplete={handleFlashComplete}
      />
      
      {appState === "results" && (
        <div className="relative">
          {error ? (
            <div className="min-h-screen flex items-center justify-center p-8">
              <div className="max-w-lg w-full bg-card border border-border rounded-xl p-6 text-center">
                <p className="text-lg font-semibold text-destructive mb-2">{error}</p>
                <p className="text-sm text-muted-foreground mb-4">Please try again or choose a different photo.</p>
                <button
                  onClick={handleNewScan}
                  className="px-6 py-3 bg-primary text-primary-foreground rounded-full font-medium hover:opacity-90 transition-opacity"
                >
                  Try Another Photo
                </button>
              </div>
            </div>
          ) : results ? (
            <ResultsView
              condition={results.condition}
              confidence={results.confidence}
              description={results.description}
            />
          ) : (
            <div className="min-h-screen flex items-center justify-center p-8">
              <div className="text-center text-muted-foreground">Preparing results...</div>
            </div>
          )}
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
