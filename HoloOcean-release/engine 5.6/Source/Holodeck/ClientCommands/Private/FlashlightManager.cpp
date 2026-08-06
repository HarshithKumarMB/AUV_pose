#include "FlashlightManager.h"

AFlashlightManager::AFlashlightManager() {
	lightColor = FLinearColor(1.0, 1.0, 1.0, 1.0);
}

void AFlashlightManager::SetLightNumber(int InLightNumber) {
	lightNumber = InLightNumber;
	UE_LOG(
		LogTemp, Log, TEXT("FlashlightManager: lightNumber set to %d"), InLightNumber);
}

void AFlashlightManager::SetLightIntensity(float InLightIntensity) {
	lightIntensity = InLightIntensity;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("FlashlightManager: lightIntensity set to %f"),
		InLightIntensity);
}

void AFlashlightManager::SetBeamWidth(float InBeamWidth) {
	beamWidth = InBeamWidth;
	UE_LOG(LogTemp, Log, TEXT("FlashlightManager: beamWidth set to %f"), InBeamWidth);
}

void AFlashlightManager::SetLightAngle(FRotator InLightAngle) {
	lightAngle = InLightAngle;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("FlashlightManager: lightAngle set to %s"),
		*InLightAngle.ToString());
}

void AFlashlightManager::SetLightPosition(FVector InLightPosition) {
	lightPosition = InLightPosition;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("FlashlightManager: lightOffset set to %s"),
		*InLightPosition.ToString());
}

void AFlashlightManager::SetLightColor(FLinearColor InLightColor) {
	lightColor = InLightColor;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("FlashlightManager: lightColor set to R:%f G:%f B:%f A:%f"),
		InLightColor.R,
		InLightColor.G,
		InLightColor.B,
		InLightColor.A);
}
