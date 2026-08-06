// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "SurfaceVesselController.h"
#include "Holodeck.h"

ASurfaceVesselController::ASurfaceVesselController(
	const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("SurfaceVessel Controller Initialized"));
}

ASurfaceVesselController::~ASurfaceVesselController() {}
