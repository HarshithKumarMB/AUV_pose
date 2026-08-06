// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "TorpedoAUVController.h"
#include "Holodeck.h"

ATorpedoAUVController::ATorpedoAUVController(
	const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("TorpedoAUV Controller Initialized"));
}

ATorpedoAUVController::~ATorpedoAUVController() {}
