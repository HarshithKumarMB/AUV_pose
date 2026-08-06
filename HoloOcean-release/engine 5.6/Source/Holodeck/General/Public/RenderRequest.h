#pragma once

#include "Holodeck.h"

#include "CoreMinimal.h"
#include "Engine/TextureRenderTarget2D.h"
#include "HolooceanPalette.h"

// Adapted from Microsoft Airsim Open Source Project
// https://github.com/Microsoft/AirSim/blob/7bddd5857791f9c164e8eba80c229f199c0babf8/Unreal/Plugins/AirSim/Source/RenderRequest.h

DECLARE_STATS_GROUP(TEXT("RenderRequest"), STATGROUP_RenderRequest, STATCAT_Advanced);
DECLARE_CYCLE_STAT_EXTERN(
	TEXT("CameraExecuteTask"),
	STAT_CameraExecuteTask,
	STATGROUP_RenderRequest, );
DECLARE_CYCLE_STAT_EXTERN(
	TEXT("CameraExecuteTask_ReadSurfaceData"),
	STAT_CameraExecuteTask_ReadSurfaceData,
	STATGROUP_RenderRequest, );
DECLARE_CYCLE_STAT_EXTERN(
	TEXT("CameraExecuteTask_Memcpy"),
	STAT_CameraExecuteTask_Memcpy,
	STATGROUP_RenderRequest, );

class FRenderRequest : public FRenderCommand {
private:
	FColor*					Buffer;
	UTextureRenderTarget2D* TargetTexture;
	HolooceanPalette		Palette;
	bool					ConvertToPalette;

public:
	virtual ~FRenderRequest();

	/**
	 * Retrieves the rendered texture from the GPU without flushing the GPU like
	 * ReadPixels() does.
	 */
	virtual void RetrievePixels(FColor* Buffer, UTextureRenderTarget2D* TargetTexture);

	// TODO: Replace this specific palette with a generic palette for extension
	virtual void RetrievePixels(
		FColor*					Buffer,
		UTextureRenderTarget2D* TargetTexture,
		bool					ConvertToPalette);

	void DoTask(
		ENamedThreads::Type	  CurrentThread,
		const FGraphEventRef& MyCompletionGraphEvent) const {
		ExecuteTask();
	}

	FORCEINLINE TStatId GetStatId() const {
		RETURN_QUICK_DECLARE_CYCLE_STAT(RenderRequest, STATGROUP_RenderThreadCommands);
	}

	void ExecuteTask() const;
};
