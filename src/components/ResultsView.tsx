import { Check } from "lucide-react";
import ProductCard from "./ProductCard";

interface ResultsViewProps {
  condition: string;
  confidence: number;
  description: string;
}

const ResultsView = ({ condition, confidence, description }: ResultsViewProps) => {
  const mockProducts = [
    {
      id: 1,
      name: "Hydrating Serum",
      brand: "DermaCare",
      price: 34.99,
      rating: 4.8,
      image: "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=400&h=400&fit=crop",
      description: "Advanced hydration formula",
    },
    {
      id: 2,
      name: "Gentle Cleanser",
      brand: "PureGlow",
      price: 24.99,
      rating: 4.6,
      image: "https://images.unsplash.com/photo-1556228578-0d85b1a4d571?w=400&h=400&fit=crop",
      description: "Perfect for sensitive skin",
    },
    {
      id: 3,
      name: "Night Repair Cream",
      brand: "SkinLux",
      price: 49.99,
      rating: 4.9,
      image: "https://images.unsplash.com/photo-1608248543803-ba4f8c70ae0b?w=400&h=400&fit=crop",
      description: "Overnight rejuvenation",
    },
  ];

  return (
    <div className="min-h-screen flex slide-in-left">
      {/* Left side - Analysis Results */}
      <div className="w-1/2 p-12 flex flex-col justify-center bg-card border-r border-border">
        <div className="max-w-lg mx-auto space-y-8">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-full bg-success flex items-center justify-center">
              <Check className="w-6 h-6 text-success-foreground" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Analysis Complete</p>
              <p className="text-lg font-semibold text-foreground">Scan successful</p>
            </div>
          </div>

          <div>
            <h2 className="text-4xl font-bold text-foreground mb-2">{condition}</h2>
            <p className="text-muted-foreground text-lg">{description}</p>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between py-4 border-b border-border">
              <span className="text-foreground font-medium">Confidence Score</span>
              <span className="text-2xl font-bold text-success">{confidence}%</span>
            </div>

            <div className="bg-secondary rounded-xl p-6 space-y-3">
              <h3 className="font-semibold text-foreground">Recommendations</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <Check className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Use gentle, fragrance-free cleansers</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Apply moisturizer twice daily</span>
                </li>
                <li className="flex items-start gap-2">
                  <Check className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                  <span>Consult a dermatologist for persistent issues</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Product Recommendations */}
      <div className="w-1/2 p-12 bg-background overflow-y-auto">
        <div className="max-w-lg mx-auto">
          <h3 className="text-2xl font-bold text-foreground mb-2">
            Recommended Products
          </h3>
          <p className="text-muted-foreground mb-8">
            Curated skincare solutions for your condition
          </p>

          <div className="space-y-4">
            {mockProducts.map((product) => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsView;
