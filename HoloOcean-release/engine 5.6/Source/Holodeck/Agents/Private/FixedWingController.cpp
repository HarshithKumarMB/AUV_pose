// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "FixedWingController.h"
#include "Holodeck.h"

AFixedWingController::AFixedWingController(const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("FixedWing Controller Initialized"));
}

AFixedWingController::~AFixedWingController() {}
