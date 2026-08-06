#pragma once

#include <cstdint>

static constexpr
#if __cplusplus >= 201703L // C++17
	inline
#endif

	uint8_t HOLOOCEAN_PALETTE_MAP[][3u] = {
		{ 0u, 0u, 0u },		  // None   			= 0u,
		{ 240u, 29u, 219u },  // Cube 			= 1u,
		{ 29u, 89u, 240u },	  // Sphere 			= 2u,
		{ 29u, 240u, 89u },	  // BaseShape 		= 3u,
		{ 153u, 153u, 153u }, // Landscape		= 4u,
		{ 91u, 235u, 52u },	  // GroundGrass 		= 5u,
		{ 196u, 101u, 6u },	  // GroundRock 		= 6u,
		{ 194u, 164u, 159u }, // Ground 			= 7u,
		{ 76u, 117u, 252u },  // GroundPath 		= 8u,
		{ 66u, 135u, 245u },  // WaterPlane      	= 9u,
		{ 128u, 64u, 128u },  // Boat				= 10u,
		{ 70u, 70u, 70u },	  // Yacht			= 11u,
		{ 102u, 102u, 156u }, // ContainerBoat	= 12u,
		{ 214u, 138u, 230u }, // Concrete       	= 13u,
		{ 73u, 227u, 78u },	  // Pipe            	= 14u,
		{ 85u, 166u, 3u },	  // PipeCover       	= 15u,
		{ 245u, 215u, 66u },  // VentCover       	= 16u,
		{ 153u, 115u, 9u },	  // Rock 			= 17u,
		{ 75u, 166u, 5u },	  // Seaweed 			= 18u,
		{ 235u, 64u, 52u },	  // Coral 			= 19u,
		{ 176u, 165u, 146u }, // Plane 			= 20u,
		{ 146u, 161u, 176u }, // Sub 				= 21u,
		{ 186u, 76u, 2u },	  // Pier				= 22u,
		{ 237u, 53u, 7u },	  // Buoy 			= 23u,
		{ 107u, 156u, 137u }, // Trash 			= 24u,
		{ 91u, 235u, 52u },	  // Grass 			= 25u,
		{ 79u, 82u, 77u },	  // Asphalt			= 26u,
		{ 212u, 191u, 125u }, // Bench			= 27u,
		{ 209u, 208u, 203u }, // BikeRack			= 28u,
		{ 31u, 240u, 205u },  // Building 		= 29u,
		{ 240u, 220u, 43u },  // Bus 				= 30u,
		{ 177u, 247u, 124u }, // Bush				= 31u,
		{ 224u, 109u, 237u }, // Car 				= 32u,
		{ 161u, 247u, 233u }, // Ceiling	 		= 33u,
		{ 65u, 13u, 255u },	  // Chair 			= 34u,
		{ 255u, 169u, 10u },  // Cone 			= 35u,
		{ 140u, 86u, 0u },	  // Crate 			= 36u,
		{ 242u, 65u, 224u },  // Desk 			= 37u,
		{ 0u, 138u, 14u },	  // Dumpster 		= 38u,
		{ 255u, 0u, 0u },	  // FireHydrant 		= 39u,
		{ 2u, 125u, 104u },	  // Floor 			= 40u,
		{ 0u, 184u, 92u },	  // GarbageCan 		= 41u,
		{ 143u, 129u, 6u },	  // Pallet 			= 42u,
		{ 222u, 164u, 177u }, // ParkingGate 		= 43u,
		{ 195u, 0u, 255u },	  // PatioUmbrella 	= 44u,
		{ 242u, 97u, 130u },  // Railing 			= 45u,
		{ 124u, 6u, 138u },	  // SemiTruck 		= 46u,
		{ 232u, 232u, 232u }, // Sidewalk 		= 47u,
		{ 222u, 218u, 245u }, // SpeedLimitSign 	= 48u,
		{ 250u, 37u, 62u },	  // StopSign 		= 49u,
		{ 255u, 149u, 10u },  // StreetLamps 		= 50u,
		{ 186u, 17u, 169u },  // Table 			= 51u,
		{ 69u, 99u, 46u },	  // Tree 			= 52u,
		{ 126u, 60u, 250u },  // Wall 			= 53u,
		{ 0u, 0u, 0u },		  // Unlabeled 		= 54u,
		{ 255u, 255u, 255u }  // Any         		= 0xFF

	};

class HolooceanPalette {
public:
	static constexpr auto GetNumberOfTags() { return std::size(HOLOOCEAN_PALETTE_MAP); }

	/// Return an RGB uint8_t array.
	///
	/// @warning It overflows if @a tag is greater than GetNumberOfTags().
	static constexpr auto GetColor(uint8_t tag) {
		return HOLOOCEAN_PALETTE_MAP[tag % GetNumberOfTags()];
	}
};