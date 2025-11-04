import { Check, Loader2, AlertCircle } from "lucide-react";
import { useEffect, useState } from "react";
import ProductCard from "./ProductCard";
import { getProductRecommendationsGemini } from "@/lib/gemini";
import { searchProducts } from "@/lib/serp_search";

interface ResultsViewProps {
  condition: string;
  confidence: number;
  description: string;
}

interface GeminiProduct {
  product: string;
  type: string;
  benefit: string;
  price: string;
}

interface Product {
  id: number;
  name: string;
  brand: string;
  price?: number;
  rating?: number;
  image?: string;
  description: string;
  link?: string;
}

const ResultsView = ({ condition, confidence, description }: ResultsViewProps) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch product recommendations from Gemini when component mounts
  useEffect(() => {
    const fetchProducts = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        console.log('🔍 Fetching product recommendations from Gemini...', { condition, confidence });
        
        // Call Gemini API
        const responseText = await getProductRecommendationsGemini(condition, confidence);
        console.log('✅ Gemini response received:', responseText);
        
        // Parse JSON response - Gemini might wrap it in markdown code blocks or return plain JSON
        let jsonText = responseText.trim();
        
        // Remove markdown code blocks if present
        if (jsonText.startsWith('```json')) {
          jsonText = jsonText.replace(/^```json\n?/, '').replace(/\n?```$/, '');
        } else if (jsonText.startsWith('```')) {
          jsonText = jsonText.replace(/^```\n?/, '').replace(/\n?```$/, '');
        }
        
        // Try to extract JSON array if there's extra text
        const jsonMatch = jsonText.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
          jsonText = jsonMatch[0];
        }
        
        // Parse the JSON
        const geminiProducts: GeminiProduct[] = JSON.parse(jsonText);
        console.log('📦 Parsed products from Gemini:', geminiProducts);
        
        // Extract product names for Serper.dev search
        const productNames = geminiProducts.map(item => item.product.trim());
        
        // Search for products using Serper.dev to get links and images
        console.log('🔍 Searching for product links and images using Serper.dev...');
        const searchResults = await searchProducts(productNames);
        
        // Transform Gemini products + Serper.dev results to our Product format
        const transformedProducts: Product[] = geminiProducts.map((item, index) => {
          // Extract brand from product name if possible (e.g., "CeraVe Foaming Cleanser" -> brand: "CeraVe")
          const nameParts = item.product.trim().split(' ');
          const brand = nameParts.length > 1 ? nameParts[0] : "Recommended";
          
          // Parse price - handle string prices like "$24.99" or "24.99"
          let parsedPrice: number | undefined;
          if (item.price && item.price.trim()) {
            const priceStr = item.price.trim().replace(/[^0-9.]/g, ''); // Remove $ and other non-numeric chars
            const priceNum = parseFloat(priceStr);
            if (!isNaN(priceNum) && priceNum > 0) {
              parsedPrice = priceNum;
            }
          }
          
          // Get link and image from Serper.dev search results
          const searchResult = searchResults[index];
          
          return {
            id: index + 1,
            name: item.product.trim(),
            brand: brand,
            description: item.benefit.trim(),
            link: searchResult?.link || undefined,
            price: parsedPrice,
            image: searchResult?.image || undefined,
            // rating is still optional - Gemini doesn't provide it
          };
        });
        
        console.log('✨ Transformed products with Serper.dev data:', transformedProducts);
        setProducts(transformedProducts);
      } catch (err: any) {
        console.error('❌ Error fetching products from Gemini:', err);
        setError(err?.message || 'Failed to load product recommendations');
      } finally {
        setIsLoading(false);
      }
    };

    fetchProducts();
  }, [condition, confidence]);

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
            AI-curated skincare solutions for your condition
          </p>

          {/* Loading State - Skeleton Cards */}
          {isLoading && (
            <div className="space-y-4">
              {/* Show 3 skeleton cards while loading */}
              {[1, 2, 3].map((index) => (
                <div
                  key={index}
                  className="bg-card rounded-2xl overflow-hidden border border-border animate-pulse"
                >
                  <div className="flex gap-4 p-4">
                    {/* Image skeleton */}
                    <div className="w-24 h-24 flex-shrink-0 rounded-xl bg-secondary/50" />
                    
                    <div className="flex-1 min-w-0 space-y-3">
                      {/* Brand and name skeleton */}
                      <div className="space-y-2">
                        <div className="h-3 w-20 bg-secondary/50 rounded" />
                        <div className="h-5 w-3/4 bg-secondary/50 rounded" />
                      </div>
                      
                      {/* Description skeleton */}
                      <div className="space-y-2">
                        <div className="h-3 w-full bg-secondary/50 rounded" />
                        <div className="h-3 w-5/6 bg-secondary/50 rounded" />
                      </div>
                      
                      {/* Price and button skeleton */}
                      <div className="flex items-center justify-between gap-2 mt-4">
                        <div className="h-6 w-16 bg-secondary/50 rounded" />
                        <div className="h-8 w-24 bg-secondary/50 rounded-full" />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              {/* Loading indicator at bottom */}
              <div className="flex flex-col items-center justify-center py-6 space-y-2">
                <Loader2 className="w-5 h-5 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">Fetching AI recommendations...</p>
              </div>
            </div>
          )}

          {/* Error State */}
          {error && !isLoading && (
            <div className="bg-destructive/10 border border-destructive/20 rounded-xl p-6 space-y-3">
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="w-5 h-5" />
                <h4 className="font-semibold">Unable to load recommendations</h4>
              </div>
              <p className="text-sm text-muted-foreground">{error}</p>
              <p className="text-xs text-muted-foreground">
                Please check your Gemini API key configuration or try again later.
              </p>
            </div>
          )}

          {/* Products List */}
          {!isLoading && !error && products.length > 0 && (
            <div className="space-y-4">
              {products.map((product) => (
                <ProductCard key={product.id} product={product} />
              ))}
            </div>
          )}

          {/* Empty State */}
          {!isLoading && !error && products.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No recommendations available at this time.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResultsView;
