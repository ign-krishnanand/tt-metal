import ttnn  
import math  
  
class TTUpsample:  
    def __init__(self, scale, num_feat, device):  
        self.scale = scale  
        self.num_feat = num_feat  
        self.device = device
        self.memory_config = ttnn.DRAM_MEMORY_CONFIG  

          
        # Pre-calculate operation parameters  
        if (scale & (scale - 1)) == 0:  # scale = 2^n  
            self.num_ops = int(math.log(scale, 2))  
            self.out_channels = 4 * num_feat  
            self.scale_factor = 2  
        elif scale == 3:  
            self.num_ops = 1  
            self.out_channels = 9 * num_feat  
            self.scale_factor = 3  
        else:  
            raise ValueError(f"Unsupported scale: {scale}")
            # Initialize compute config for the device  
        self.compute_config = ttnn.init_device_compute_kernel_config(  
            device.arch(),  
            math_fidelity=ttnn.MathFidelity.LoFi,  
            fp32_dest_acc_en=False,  
            packer_l1_acc=False,  
        )  
        # Initialize conv config with no activation and default output layout  
        self.conv_config = ttnn.Conv2dConfig(  
            weights_dtype=ttnn.bfloat16,  
            activation="",  
            output_layout=ttnn.TILE_LAYOUT,  
            deallocate_activation=True,  # Free input memory after use  
            reallocate_halo_output=True,  # Reduce memory fragmentation  
            act_block_h_override=32,  # Use smaller activation blocks  
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,  # Use height sharding  
        )
   
    def pixel_shuffle(self, x, upscale_factor):  
        """Implement PixelShuffle operation in TTNN"""  
        # Convert to interleaved layout before reshape to avoid sharding issues  
        if x.is_sharded():  
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  
        
        # Ensure we're in ROW_MAJOR layout for reshape operations  
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:  
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)  
        
        # x shape: [B, H, W, C] where C = input_channels * upscale_factor^2  
        batch_size, height, width, channels = x.shape  
        out_channels = channels // (upscale_factor * upscale_factor)  
        
        # Reshape to separate the upscale dimensions - CORRECT NHWC ORDER  
        # [B, H, W, out_channels, upscale_factor, upscale_factor]  
        x = ttnn.reshape(x, (batch_size, height, width, out_channels, upscale_factor, upscale_factor))  
        
        # Permute to rearrange dimensions for upsampling  
        # [B, H, upscale_factor, W, upscale_factor, out_channels]  
        x = ttnn.permute(x, (0, 1, 4, 2, 5, 3))  
        
        # Reshape to final upsampled size  
        # [B, H*upscale_factor, W*upscale_factor, out_channels]  
        output_height = height * upscale_factor  
        output_width = width * upscale_factor  
        x = ttnn.reshape(x, (batch_size, output_height, output_width, out_channels))  
        
        return x

    def __call__(self, x, weights, bias=None):  
        current = x  
        current_channels = self.num_feat  # Start with 4 channels  
        
        for i in range(self.num_ops):  
            # Calculate output channels for this specific convolution  
            out_channels = current_channels * (self.scale_factor * self.scale_factor)
            batch_size=current.shape[0]
            height = current.shape[1]  
            width = current.shape[2]  
            current, (out_height, out_width) = ttnn.conv2d(  
                input_tensor=current,  
                weight_tensor=weights[i],  
                bias_tensor=bias[i] if bias else None,  
                in_channels=current_channels,  # Use dynamic channel count  
                out_channels=out_channels,     # Use calculated output channels  
                device=self.device,  
                kernel_size=(3, 3),  
                stride=(1, 1),  
                padding=(1, 1),  
                batch_size=batch_size,  
                input_height=height,  
                input_width=width,  
                conv_config=self.conv_config,  
                compute_config=self.compute_config,  
                dtype=ttnn.bfloat16,
                return_output_dim=True,  # Only return the output tensor for simplest call  
            )
            # reshape B,H*W, C to B,1,H*W, C
            current = ttnn.reshape(current, (batch_size, 1,  out_height*out_width, out_channels))  
            print(current.shape)  
            current = ttnn.to_layout(current, ttnn.ROW_MAJOR_LAYOUT)
            # reshape B,1,H*W, C to B, H, W, C
            current = ttnn.reshape(current, (batch_size,  current.shape[2]//height, current.shape[2]//height, out_channels))
            # Apply pixel shuffle  
            current = self.pixel_shuffle(current, self.scale_factor)  
            
            # After pixel shuffle, channels return to original count  
            current_channels = self.num_feat  
        
        # Convert to NCHW format  
        current = ttnn.permute(current, (0, 3, 1, 2), memory_config=self.memory_config)  
        
        return current