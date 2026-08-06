// MIT License (c) 2021 BYU FRoStLab see LICENSE file

#include "BlueROV2Controller.h"
#include "Holodeck.h"

ABlueROV2Controller::ABlueROV2Controller(const FObjectInitializer& ObjectInitializer)
	: AHolodeckPawnController(ObjectInitializer) {
	UE_LOG(LogTemp, Warning, TEXT("BlueROV2 Controller Initialized"));
}

ABlueROV2Controller::~ABlueROV2Controller() {}
