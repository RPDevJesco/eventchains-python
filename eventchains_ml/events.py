"""
Machine Learning Events for EventChains

These events provide discrete, observable steps in the ML training pipeline.
Each event handles one specific aspect of training, making the process transparent and debuggable.
"""

import torch
from eventchains import ChainableEvent, Result

class LoadBatchEvent(ChainableEvent):
    """
    Load a batch of data from the dataloader.
    
    Sets in context:
        - 'batch': Input data tensor
        - 'labels': Target labels tensor
        - 'batch_size': Size of the current batch
    """
    
    def __init__(self, dataloader):
        """
        Initialize the LoadBatchEvent.
        
        Args:
            dataloader: PyTorch DataLoader instance
        """
        self.dataloader = dataloader
        self.iterator = None
    
    def execute(self, context):
        # Initialize iterator if needed
        if self.iterator is None:
            self.iterator = iter(self.dataloader)
        
        try:
            batch, labels = next(self.iterator)
            
            # Move to device if specified
            device = context.get('device', 'cpu')
            batch = batch.to(device)
            labels = labels.to(device)
            
            context.set('batch', batch)
            context.set('labels', labels)
            context.set('batch_size', len(batch))
            
            return Result.ok()
        
        except StopIteration:
            # Reset iterator for next epoch
            self.iterator = None
            return Result.fail('No more batches in dataloader')

class ForwardPassEvent(ChainableEvent):
    """
    Execute forward pass through the model.
    
    Reads from context:
        - 'batch': Input data tensor
        - 'model': PyTorch model
    
    Sets in context:
        - 'output': Model output tensor
        - 'layer_activations': Dict of activation statistics per layer (if track_activations=True)
    """
    
    def __init__(self, track_activations=True):
        """
        Initialize the ForwardPassEvent.
        
        Args:
            track_activations: Whether to track activation statistics (default: True)
        """
        self.track_activations = track_activations
        self.activation_hooks = []
        self.activations = {}
    
    def execute(self, context):
        batch = context.get('batch')
        model = context.get('model')
        
        if batch is None:
            return Result.fail('No batch in context')
        if model is None:
            return Result.fail('No model in context')
        
        # Clear previous activations
        self.activations.clear()
        
        # Register hooks to capture activations if needed
        if self.track_activations:
            self._register_hooks(model)
        
        # Forward pass
        try:
            output = model(batch)
            context.set('output', output)
            
            # Store activation statistics
            if self.track_activations:
                context.set('layer_activations', self._compute_activation_stats())
                self._remove_hooks()
            
            return Result.ok()
        
        except Exception as e:
            self._remove_hooks()
            return Result.fail(f'Forward pass failed: {str(e)}')
    
    def _register_hooks(self, model):
        """Register forward hooks to capture activations."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
            return hook
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(hook_fn(name))
                self.activation_hooks.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.activation_hooks:
            handle.remove()
        self.activation_hooks.clear()
    
    def _compute_activation_stats(self):
        """Compute statistics for captured activations."""
        stats = {}
        for name, activation in self.activations.items():
            if activation.numel() > 0:
                stats[name] = {
                    'mean': activation.mean().item(),
                    'std': activation.std().item(),
                    'min': activation.min().item(),
                    'max': activation.max().item(),
                    'zeros': (activation == 0).sum().item(),
                    'total': activation.numel(),
                    'zero_percent': ((activation == 0).sum().item() / activation.numel() * 100),
                }
        return stats

class CalculateLossEvent(ChainableEvent):
    """
    Calculate loss between model output and target labels.
    
    Reads from context:
        - 'output': Model output tensor
        - 'labels': Target labels tensor
        - 'criterion': Loss function
    
    Sets in context:
        - 'loss': Computed loss value (tensor)
        - 'loss_value': Loss as Python float
    """
    
    def __init__(self, criterion=None):
        """
        Initialize the CalculateLossEvent.
        
        Args:
            criterion: Loss function (optional, can be provided in context)
        """
        self.criterion = criterion
    
    def execute(self, context):
        output = context.get('output')
        labels = context.get('labels')
        criterion = self.criterion or context.get('criterion')
        
        if output is None:
            return Result.fail('No output in context')
        if labels is None:
            return Result.fail('No labels in context')
        if criterion is None:
            return Result.fail('No criterion provided')
        
        try:
            loss = criterion(output, labels)
            context.set('loss', loss)
            context.set('loss_value', loss.item())
            
            return Result.ok()
        
        except Exception as e:
            return Result.fail(f'Loss calculation failed: {str(e)}')

class BackpropagationEvent(ChainableEvent):
    """
    Perform backpropagation to compute gradients.
    
    Reads from context:
        - 'loss': Loss tensor
        - 'model': PyTorch model
    
    Sets in context:
        - 'gradient_norms': Dict of gradient norms per parameter
        - 'total_gradient_norm': Total gradient norm across all parameters
    """
    
    def __init__(self, track_gradients=True):
        """
        Initialize the BackpropagationEvent.
        
        Args:
            track_gradients: Whether to track gradient statistics (default: True)
        """
        self.track_gradients = track_gradients
    
    def execute(self, context):
        loss = context.get('loss')
        model = context.get('model')
        
        if loss is None:
            return Result.fail('No loss in context')
        if model is None:
            return Result.fail('No model in context')
        
        try:
            # Perform backpropagation
            loss.backward()
            
            # Track gradient statistics if requested
            if self.track_gradients:
                gradient_norms = {}
                total_norm = 0.0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        gradient_norms[name] = grad_norm
                        total_norm += grad_norm ** 2
                
                total_norm = total_norm ** 0.5
                
                context.set('gradient_norms', gradient_norms)
                context.set('total_gradient_norm', total_norm)
            
            return Result.ok()
        
        except Exception as e:
            return Result.fail(f'Backpropagation failed: {str(e)}')

class UpdateWeightsEvent(ChainableEvent):
    """
    Update model weights using the optimizer.
    
    Reads from context:
        - 'optimizer': PyTorch optimizer
    
    Sets in context:
        - 'learning_rate': Current learning rate
    """
    
    def __init__(self, optimizer=None):
        """
        Initialize the UpdateWeightsEvent.
        
        Args:
            optimizer: PyTorch optimizer (optional, can be provided in context)
        """
        self.optimizer = optimizer
    
    def execute(self, context):
        optimizer = self.optimizer or context.get('optimizer')
        
        if optimizer is None:
            return Result.fail('No optimizer provided')
        
        try:
            # Update weights
            optimizer.step()
            
            # Store current learning rate
            lr = optimizer.param_groups[0]['lr']
            context.set('learning_rate', lr)
            
            # Zero gradients for next iteration
            optimizer.zero_grad()
            
            return Result.ok()
        
        except Exception as e:
            return Result.fail(f'Weight update failed: {str(e)}')

class ValidationEvent(ChainableEvent):
    """
    Perform validation on a validation dataset.
    
    Reads from context:
        - 'model': PyTorch model
        - 'val_dataloader': Validation dataloader
        - 'criterion': Loss function
    
    Sets in context:
        - 'val_loss': Average validation loss
        - 'val_accuracy': Validation accuracy (if classification)
    """
    
    def __init__(self, val_dataloader=None, compute_accuracy=True):
        """
        Initialize the ValidationEvent.
        
        Args:
            val_dataloader: Validation dataloader (optional, can be in context)
            compute_accuracy: Whether to compute accuracy for classification (default: True)
        """
        self.val_dataloader = val_dataloader
        self.compute_accuracy = compute_accuracy
    
    def execute(self, context):
        model = context.get('model')
        val_dataloader = self.val_dataloader or context.get('val_dataloader')
        criterion = context.get('criterion')
        device = context.get('device', 'cpu')
        
        if model is None:
            return Result.fail('No model in context')
        if val_dataloader is None:
            return Result.fail('No validation dataloader provided')
        if criterion is None:
            return Result.fail('No criterion in context')
        
        try:
            model.eval()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch, labels in val_dataloader:
                    batch = batch.to(device)
                    labels = labels.to(device)
                    
                    output = model(batch)
                    loss = criterion(output, labels)
                    
                    total_loss += loss.item()
                    
                    if self.compute_accuracy:
                        _, predicted = output.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
            
            avg_loss = total_loss / len(val_dataloader)
            context.set('val_loss', avg_loss)
            
            if self.compute_accuracy:
                accuracy = 100.0 * correct / total
                context.set('val_accuracy', accuracy)
            
            model.train()
            
            return Result.ok()
        
        except Exception as e:
            model.train()
            return Result.fail(f'Validation failed: {str(e)}')

class TokenCandidateEvent(ChainableEvent):
    """Generate next token candidate using policy"""

    def __init__(self, policy):
        self.policy = policy

    def execute(self, context):
        candidate = self.policy.next_candidate()
        if candidate is None:
            return Result.fail("No more candidates")
        context.set('candidate', candidate)
        return Result.ok()

class ForwardPassInversionEvent(ChainableEvent):
    """Run forward pass for candidate token"""

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx

    def execute(self, context):
        candidate = context.get('candidate')
        prefix = context.get('prefix', [])

        # Forward pass to target layer
        with torch.no_grad():
            hidden = self.model.forward_to_layer(
                prefix + [candidate],
                self.layer_idx
            )

        context.set('predicted_hidden', hidden)
        return Result.ok()

class VerifyAcceptanceEvent(ChainableEvent):
    """Verify if candidate is within acceptance region"""

    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def execute(self, context):
        predicted = context.get('predicted_hidden')
        target = context.get('target_hidden')

        distance = torch.norm(predicted - target).item()

        context.set('distance', distance)

        if distance < self.epsilon:
            context.set('verified', True)
            context.set('recovered_token', context.get('candidate'))
            return Result.ok()  # Signal to stop

        context.set('verified', False)
        return Result.ok()  # Continue searching

class CollisionDetectionEvent(ChainableEvent):
    """Test for collisions in token representations"""

    def execute(self, context):
        model = context.get('model')
        vocabulary = context.get('vocabulary')
        layer_idx = context.get('layer_idx')
        prefix = context.get('prefix', [])

        representations = {}
        collisions = []

        with torch.no_grad():
            for token in vocabulary:
                hidden = model.forward_to_layer(
                    prefix + [token],
                    layer_idx
                )
                hidden_key = tuple(hidden.cpu().numpy().round(6))

                if hidden_key in representations:
                    collisions.append({
                        'token1': representations[hidden_key],
                        'token2': token,
                        'hidden': hidden
                    })
                else:
                    representations[hidden_key] = token

        context.set('num_collisions', len(collisions))
        context.set('collisions', collisions)

        return Result.ok()