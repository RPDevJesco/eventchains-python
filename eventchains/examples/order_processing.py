"""
Order processing example demonstrating business workflow with EventChains.
"""

from eventchains import EventChain, ChainableEvent, EventContext, Result, Middleware, FaultTolerance


# Events
class ValidateOrder(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        
        if not order:
            return Result.fail("Order is missing")
        
        if not order.get('items'):
            return Result.fail("Order has no items")
        
        if not order.get('customer_id'):
            return Result.fail("Customer ID is missing")
        
        print(f"✓ Order validated for customer {order['customer_id']}")
        return Result.ok()


class CalculateTotals(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        items = order['items']
        
        subtotal = sum(item['price'] * item['quantity'] for item in items)
        tax = subtotal * 0.08  # 8% tax
        total = subtotal + tax
        
        context.set('subtotal', subtotal)
        context.set('tax', tax)
        context.set('total', total)
        
        print(f"✓ Calculated totals - Subtotal: ${subtotal:.2f}, Tax: ${tax:.2f}, Total: ${total:.2f}")
        return Result.ok()


class CheckInventory(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        items = order['items']
        
        # Simulate inventory check
        for item in items:
            print(f"  Checking inventory for {item['name']}...")
        
        # Assume all items are in stock
        context.set('inventory_available', True)
        print(f"✓ Inventory check passed")
        return Result.ok()


class ProcessPayment(ChainableEvent):
    def execute(self, context):
        total = context.get('total')
        order = context.get('order')
        
        # Simulate payment processing
        payment_method = order.get('payment_method', 'credit_card')
        payment_id = f"PAY-{hash(str(total)) % 100000:05d}"
        
        context.set('payment_id', payment_id)
        context.set('payment_status', 'completed')
        
        print(f"✓ Payment processed: {payment_id} (${total:.2f} via {payment_method})")
        return Result.ok()


class CreateShipment(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        
        # Simulate shipment creation
        shipment_id = f"SHIP-{hash(str(order['customer_id'])) % 100000:05d}"
        tracking_number = f"TRACK-{shipment_id[-5:]}-{hash(str(order)) % 10000:04d}"
        
        context.set('shipment_id', shipment_id)
        context.set('tracking_number', tracking_number)
        
        print(f"✓ Shipment created: {shipment_id} (Tracking: {tracking_number})")
        return Result.ok()


class SendConfirmationEmail(ChainableEvent):
    def execute(self, context):
        order = context.get('order')
        payment_id = context.get('payment_id')
        tracking_number = context.get('tracking_number')
        total = context.get('total')
        
        # Simulate email sending
        customer_email = order.get('customer_email', 'customer@example.com')
        
        print(f"✓ Confirmation email sent to {customer_email}")
        print(f"  Order Total: ${total:.2f}")
        print(f"  Payment ID: {payment_id}")
        print(f"  Tracking: {tracking_number}")
        
        return Result.ok()


# Middleware
class OrderLoggingMiddleware(Middleware):
    def execute(self, context, next_callable):
        event_name = context.get('_current_event', 'Unknown')
        order_id = context.get('order', {}).get('id', 'N/A')
        
        print(f"\n[{order_id}] → {event_name}")
        result = next_callable(context)
        
        if not result.success:
            print(f"[{order_id}] ✗ {event_name} failed: {result.error}")
        
        return result


class PerformanceMonitorMiddleware(Middleware):
    def __init__(self):
        self.timings = {}
    
    def execute(self, context, next_callable):
        import time
        event_name = context.get('_current_event', 'Unknown')
        
        start = time.perf_counter()
        result = next_callable(context)
        elapsed = (time.perf_counter() - start) * 1000
        
        if event_name not in self.timings:
            self.timings[event_name] = []
        self.timings[event_name].append(elapsed)
        
        return result
    
    def report(self):
        print("\n" + "=" * 60)
        print("Performance Report")
        print("=" * 60)
        for event, times in sorted(self.timings.items()):
            avg = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            print(f"{event:30} avg: {avg:6.2f}ms  min: {min_time:6.2f}ms  max: {max_time:6.2f}ms  calls: {len(times)}")


def create_sample_order(order_id, customer_id):
    return {
        'id': order_id,
        'customer_id': customer_id,
        'customer_email': f'customer{customer_id}@example.com',
        'payment_method': 'credit_card',
        'items': [
            {'name': 'Widget', 'price': 19.99, 'quantity': 2},
            {'name': 'Gadget', 'price': 49.99, 'quantity': 1},
            {'name': 'Doohickey', 'price': 9.99, 'quantity': 3}
        ]
    }


def main():
    print("=" * 60)
    print("EventChains Order Processing Example")
    print("=" * 60)
    
    # Create middleware instances
    performance_monitor = PerformanceMonitorMiddleware()
    
    # Build the order processing chain
    order_chain = (EventChain(fault_tolerance=FaultTolerance.STRICT)
        .add_event(ValidateOrder())
        .add_event(CalculateTotals())
        .add_event(CheckInventory())
        .add_event(ProcessPayment())
        .add_event(CreateShipment())
        .add_event(SendConfirmationEmail())
        .use_middleware(OrderLoggingMiddleware())
        .use_middleware(performance_monitor))
    
    print(f"\nChain: {order_chain}\n")
    
    # Process multiple orders
    orders = [
        create_sample_order('ORD-001', 'CUST-123'),
        create_sample_order('ORD-002', 'CUST-456'),
        create_sample_order('ORD-003', 'CUST-789'),
    ]
    
    successful = 0
    failed = 0
    
    for order in orders:
        context = EventContext({'order': order})
        result = order_chain.execute(context)
        
        if result.success:
            successful += 1
            print(f"\n✓ Order {order['id']} processed successfully")
        else:
            failed += 1
            print(f"\n✗ Order {order['id']} failed: {result.error}")
    
    # Show performance report
    performance_monitor.report()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {successful} successful, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
