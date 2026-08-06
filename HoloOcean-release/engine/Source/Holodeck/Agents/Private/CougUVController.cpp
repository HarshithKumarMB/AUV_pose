// MIT License (c) 2019 BYU PCCL see LICENSE file

#include "CougUVController.h"
#include "Holodeck.h"

ACougUVController::ACougUVController(const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("CougUV Controller Initialized"));
}

ACougUVController::~ACougUVController() {}
