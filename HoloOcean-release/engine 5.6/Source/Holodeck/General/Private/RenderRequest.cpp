#include "RenderRequest.h"
#include "Holodeck.h"

DEFINE_STAT(STAT_CameraExecuteTask);
DEFINE_STAT(STAT_CameraExecuteTask_ReadSurfaceData);
DEFINE_STAT(STAT_CameraExecuteTask_Memcpy);

void FRenderRequest::RetrievePixels(
	FColor*					PixelBuffer,
	UTextureRenderTarget2D* Texture) {
	this->Buffer = PixelBuffer;
	this->TargetTexture = Texture;
	CheckNotBlockedOnRenderThread(); // not the issue

	// Queue up the task of rendering the scene in the render thread
	TGraphTask<FRenderRequest>::CreateTask().ConstructAndDispatchWhenReady(*this);
}

void FRenderRequest::RetrievePixels(
	FColor*					PixelBuffer,
	UTextureRenderTarget2D* Texture,
	bool					ShouldConvertToPalette) {
	this->Buffer = PixelBuffer;
	this->TargetTexture = Texture;
	this->ConvertToPalette = ShouldConvertToPalette;
	this->Palette = HolooceanPalette();
	CheckNotBlockedOnRenderThread(); // not the issue

	// Queue up the task of rendering the scene in the render thread
	TGraphTask<FRenderRequest>::CreateTask().ConstructAndDispatchWhenReady(*this);
}

void FRenderRequest::ExecuteTask() const {
	SCOPE_CYCLE_COUNTER(STAT_CameraExecuteTask);

	TArray<FColor>			  SurfaceData;
	FRHICommandListImmediate& RHICmdList = GetImmediateCommandList_ForRenderCommand();
	FTextureRenderTargetResource* rt_resource =
		TargetTexture->GetRenderTargetResource();

	if (rt_resource != nullptr) {
		const FTexture2DRHIRef& rhi_texture = rt_resource->GetRenderTargetTexture();
		// FIntPoint size;
		FReadSurfaceDataFlags	flags(RCM_UNorm, CubeFace_MAX);
		flags.SetLinearToGamma(false);

		{
			SCOPE_CYCLE_COUNTER(STAT_CameraExecuteTask_ReadSurfaceData);
			// This next call is slow! Significant impact on the frame time (~8ms)
			RHICmdList.ReadSurfaceData(
				rhi_texture,
				FIntRect(0, 0, TargetTexture->SizeX, TargetTexture->SizeY),
				SurfaceData,
				flags);
		}

		{
			SCOPE_CYCLE_COUNTER(STAT_CameraExecuteTask_Memcpy);

			if (ConvertToPalette) {
				// Translate the surface data using the palette
				for (int i = 0; i < SurfaceData.Num(); i++) {
					// UE_LOG(LogHolodeck, Warning, TEXT("%d"), SurfaceData[i].R);
					const unsigned char* color = Palette.GetColor(SurfaceData[i].R);
					SurfaceData[i] = FColor(color[0], color[1], color[2]);
				}
			}

			FMemory::Memcpy(
				this->Buffer,
				&SurfaceData[0],
				SurfaceData.Num() * sizeof(FColor)); // this line isn't the problem
		}
	}
}

FRenderRequest::~FRenderRequest() {};
