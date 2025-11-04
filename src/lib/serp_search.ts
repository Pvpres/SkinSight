/**
 * Serper.dev Search API integration
 * Uses Serper.dev to search for products and get their links, images, prices, and ratings
 * 
 * API Documentation: https://serper.dev/
 */

// Get API key from environment
const getApiKey = (): string | undefined => {
    if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_SERPER_SEARCH_API_KEY) {
      return import.meta.env.VITE_SERPER_SEARCH_API_KEY.trim();
    }
    if (typeof process !== 'undefined') {
      const key = process.env?.SERPER_SEARCH_API_KEY || process.env?.VITE_SERPER_SEARCH_API_KEY;
      return key?.trim();
    }
    return undefined;
  };
  
  interface SerperSearchResult {
    link: string;
    image?: string;
    price?: number;
    rating?: number;
  }
  
  /**
   * Extract result data from Serper API response
   */
  function extractResult(result: any): { link: string; image?: string; price?: number; rating?: number } | null {
    if (!result) return null;
    
    const link = result.link || '';
    if (!link) return null;
    
    return {
      link,
      image: result.image || result.thumbnail || result.imageUrl,
      price: result.price,
      rating: result.rating,
    };
  }
  
  /**
   * Search for product images using Serper.dev image search
   */
  async function searchProductImage(productName: string): Promise<string | undefined> {
    const apiKey = getApiKey();
    if (!apiKey) return undefined;
  
    try {
      const response = await fetch('https://google.serper.dev/images', {
        method: 'POST',
        headers: { 'X-API-KEY': apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: productName, gl: 'us', hl: 'en', num: 5 }),
      });
  
      if (!response.ok) return undefined;
      const data = await response.json();
      return data.images?.[0]?.imageUrl;
    } catch (error) {
      return undefined;
    }
  }
  
  /**
   * Search for a product using Serper.dev API
   */
  export async function searchProduct(
    productName: string,
    skipImageFallback: boolean = false
  ): Promise<SerperSearchResult | null> {
    const apiKey = getApiKey();
    
    if (!apiKey) {
      console.warn('⚠️ SERPER_SEARCH_API_KEY not found');
      return null;
    }
  
    try {
      const response = await fetch('https://google.serper.dev/search', {
        method: 'POST',
        headers: { 'X-API-KEY': apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q: `${productName} buy amazon sephora ulta`,
          gl: 'us',
          hl: 'en',
          num: 10,
        }),
      });
  
      if (!response.ok) return null;
      const data = await response.json();
      
      // Try shopping, organic, then images
      let result = extractResult(data.shopping?.[0]) 
                || extractResult(data.organic?.[0]) 
                || extractResult(data.images?.[0]);
      
      if (!result) return null;
      
      // Fallback image search if needed
      if (!result.image && !skipImageFallback) {
        result.image = await searchProductImage(productName);
      }
      
      return result;
    } catch (error) {
      return null;
    }
  }
  
  /**
   * Search for multiple products (batch optimized)
   */
  export async function searchProducts(
    productNames: string[]
  ): Promise<(SerperSearchResult | null)[]> {
    console.log(`🔍 Searching for ${productNames.length} products...`);
    
    // Step 1: Search all products (skip image fallback)
    const searchPromises = productNames.map((name, i) => 
      new Promise<SerperSearchResult | null>(resolve => 
        setTimeout(() => resolve(searchProduct(name, true)), i * 100)
      )
    );
    const results = await Promise.all(searchPromises);
    
    const successCount = results.filter(r => r !== null).length;
    console.log(`✅ Found ${successCount}/${productNames.length} products`);
    
    // Step 2: Batch search missing images
    const needImages = results
      .map((r, i) => ({ result: r, index: i, name: productNames[i] }))
      .filter(({ result }) => result && !result.image);
    
    if (needImages.length > 0) {
      console.log(`🖼️ Searching for ${needImages.length} missing images...`);
      
      const imagePromises = needImages.map(({ name }, i) => 
        new Promise<string | undefined>(resolve => 
          setTimeout(() => resolve(searchProductImage(name)), i * 100)
        )
      );
      const images = await Promise.all(imagePromises);
      
      images.forEach((img, i) => {
        if (img && results[needImages[i].index]) {
          results[needImages[i].index]!.image = img;
        }
      });
    }
    
    // Summary
    const counts = {
      images: results.filter(r => r?.image).length,
      prices: results.filter(r => r?.price).length,
      ratings: results.filter(r => r?.rating).length,
    };
    console.log(`📊 Results: ${successCount} links, ${counts.images} images, ${counts.prices} prices, ${counts.ratings} ratings`);
    
    return results;
  }