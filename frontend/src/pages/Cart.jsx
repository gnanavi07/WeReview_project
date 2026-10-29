
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { ShoppingCart, Trash2, Plus, Minus } from 'lucide-react'

export default function Cart() {
  const [cart, setCart] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('wereview_cart') || '[]')
    } catch {
      return []
    }
  })

  const saveCart = (items) => {
    setCart(items)
    localStorage.setItem('wereview_cart', JSON.stringify(items))
  }

  const updateQuantity = (id, change) => {
    saveCart(
      cart
        .map(item =>
          item.id === id
            ? { ...item, quantity: item.quantity + change }
            : item
        )
        .filter(item => item.quantity > 0)
    )
  }

  const removeItem = (id) => {
    saveCart(cart.filter(item => item.id !== id))
  }

  const total = cart.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  )

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Shopping Cart</h1>

      {cart.length === 0 ? (
        <div className="bg-white rounded-xl p-10 text-center border">
          <ShoppingCart size={48} className="mx-auto mb-4 text-gray-400" />
          <p className="text-gray-600 mb-5">Your cart is empty.</p>
          <Link
            to="/shop"
            className="inline-block bg-purple-600 text-white px-5 py-2 rounded-lg"
          >
            Continue Shopping
          </Link>
        </div>
      ) : (
        <>
          <div className="space-y-4">
            {cart.map(item => (
              <div key={item.id} className="bg-white border rounded-xl p-4 flex items-center gap-4">
                <img
                  src={item.image}
                  alt={item.name}
                  className="w-20 h-20 object-cover rounded-lg"
                />

                <div className="flex-1">
                  <h2 className="font-semibold">{item.name}</h2>
                  <p className="text-purple-700 font-bold">
                    ₹{item.price.toLocaleString('en-IN')}
                  </p>

                  <div className="flex items-center gap-3 mt-2">
                    <button
                      onClick={() => updateQuantity(item.id, -1)}
                      className="border rounded p-1"
                      aria-label="Decrease quantity"
                    >
                      <Minus size={16} />
                    </button>
                    <span>{item.quantity}</span>
                    <button
                      onClick={() => updateQuantity(item.id, 1)}
                      className="border rounded p-1"
                      aria-label="Increase quantity"
                    >
                      <Plus size={16} />
                    </button>
                  </div>
                </div>

                <button
                  onClick={() => removeItem(item.id)}
                  className="text-red-600 p-2"
                  aria-label={`Remove ${item.name}`}
                >
                  <Trash2 size={20} />
                </button>
              </div>
            ))}
          </div>

          <div className="bg-white border rounded-xl p-5 mt-6">
            <p className="text-lg font-bold">
              Total: ₹{total.toLocaleString('en-IN')}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Checkout is not connected yet.
            </p>
          </div>
        </>
      )}
    </div>
  )
}
