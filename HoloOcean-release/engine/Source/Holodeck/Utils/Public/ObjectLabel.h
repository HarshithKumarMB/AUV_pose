// Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <cstdint>

namespace holoocean {

	enum class ObjectLabel : uint8_t {
		None = 0u,
		Cube = 1u,
		Sphere = 2u,
		BaseShape = 3u,
		Landscape = 4u,
		GroundGrass = 5u,
		GroundRock = 6u,
		Ground = 7u,
		GroundPath = 8u,
		WaterPlane = 9u,
		Boat = 10u,
		Yacht = 11u,
		ContainerBoat = 12u,
		Concrete = 13u,
		Pipe = 14u,
		PipeCover = 15u,
		VentCover = 16u,
		Rock = 17u,
		Seaweed = 18u,
		Coral = 19u,
		Plane = 20u,
		Sub = 21u,
		Pier = 22u,
		Buoy = 23u,
		Trash = 24u,
		Grass = 25u,
		Asphalt = 26u,
		Bench = 27u,
		BikeRack = 28u,
		Building = 29u,
		Bus = 30u,
		Bush = 31u,
		Car = 32u,
		Ceiling = 33u,
		Chair = 34u,
		Cone = 35u,
		Crate = 36u,
		Desk = 37u,
		Dumpster = 38u,
		FireHydrant = 39u,
		Floor = 40u,
		GarbageCan = 41u,
		Pallet = 42u,
		ParkingGate = 43u,
		PatioUmbrella = 44u,
		Railing = 45u,
		SemiTruck = 46u,
		Sidewalk = 47u,
		SpeedLimitSign = 48u,
		StopSign = 49u,
		StreetLamps = 50u,
		Table = 51u,
		Tree = 52u,
		Wall = 53u,
		Unlabeled = 54u,
		Any = 0xFF
	};
} // namespace holoocean