import { Star, ShoppingCart } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Product {
  id: number;
  name: string;
  brand: string;
  price: number;
  rating: number;
  image: string;
  description: string;
}

interface ProductCardProps {
  product: Product;
}

const ProductCard = ({ product }: ProductCardProps) => {
  return (
    <div className="bg-card rounded-2xl overflow-hidden border border-border hover:shadow-lg transition-all duration-300 hover:scale-[1.02]">
      <div className="flex gap-4 p-4">
        <div className="w-24 h-24 flex-shrink-0 rounded-xl overflow-hidden bg-secondary">
          <img
            src={product.image}
            alt={product.name}
            className="w-full h-full object-cover"
          />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2 mb-1">
            <div className="flex-1 min-w-0">
              <p className="text-xs text-muted-foreground">{product.brand}</p>
              <h4 className="font-semibold text-foreground truncate">{product.name}</h4>
            </div>
            <div className="flex items-center gap-1 bg-secondary px-2 py-1 rounded-lg flex-shrink-0">
              <Star className="w-3 h-3 fill-accent text-accent" />
              <span className="text-xs font-medium">{product.rating}</span>
            </div>
          </div>

          <p className="text-sm text-muted-foreground mb-3 line-clamp-1">
            {product.description}
          </p>

          <div className="flex items-center justify-between gap-2">
            <span className="text-xl font-bold text-foreground">
              ${product.price}
            </span>
            <Button size="sm" className="gap-2">
              <ShoppingCart className="w-4 h-4" />
              Add
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductCard;
