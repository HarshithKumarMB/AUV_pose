#include "WeatherManager.h"

AWeatherManager::AWeatherManager() {
	bIsRainy = false;
	rainVelocity = FVector(0, 300, -1000);
	spawnRate = 3000;
}

void AWeatherManager::SetIsRainy(bool bRain) {
	bIsRainy = bRain;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("WeatherManager: bIsRainy set to %s"),
		bRain ? TEXT("true") : TEXT("false"));
}

void AWeatherManager::SetRainVelocity(FVector newRainVelocity) {
	rainVelocity = newRainVelocity;
	UE_LOG(
		LogTemp,
		Log,
		TEXT("WeatherManager: rainVelocity set to %s"),
		*rainVelocity.ToString());
}

void AWeatherManager::SetSpawnRate(float newSpawnRate) {
	spawnRate = newSpawnRate;
	UE_LOG(LogTemp, Log, TEXT("WeatherManager: spawnRate set to %f"), newSpawnRate);
}