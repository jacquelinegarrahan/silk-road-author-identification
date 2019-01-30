import matplotlib.pyplot as plt


Tw_25_hidden_25_traccuracy =[0.1756, 0.1911, 0.1892, 0.1892, 0.1915, 0.1919, 0.1985, 0.1946, 0.1974, 0.1931, 0.2032, 0.1974, 0.1977, 0.2020, 0.2009, 0.2012, 0.1974, 0.1958, 0.2016, 0.2098, 0.2114, 0.2207, 0.2149, 0.2195, 0.2164, 0.2238, 0.2199, 0.2172, 0.2301, 0.2293, 0.2238, 0.2285, 0.2328, 0.2320, 0.2262, 0.2308, 0.2371, 0.2386, 0.2308, 0.2402, 0.2425, 0.2339, 0.2429, 0.2367, 0.2363, 0.2526, 0.2390, 0.2448, 0.2406, 0.2456, 0.2421, 0.2565, 0.2534, 0.2542, 0.2526, 0.2487, 0.2515, 0.2511, 0.2491, 0.2557, 0.2651, 0.2569, 0.2592, 0.2577, 0.2616, 0.2550, 0.2596, 0.2663, 0.2592, 0.2589, 0.2561, 0.2659, 0.2589, 0.2663, 0.2694, 0.2678, 0.2764, 0.2775, 0.2795, 0.2799, 0.2877, 0.2896, 0.2826, 0.2982, 0.2943, 0.2896, 0.2966, 0.3032, 0.2989, 0.3134, 0.3134, 0.3013, 0.3126, 0.3122, 0.3040, 0.3071, 0.3075, 0.3106, 0.3184]
Tw_25_hidden_25_valaccuracy =[0.2060, 0.1978, 0.1978, 0.1978, 0.1978, 0.2042, 0.2083, 0.2071, 0.2071, 0.2089, 0.2077, 0.2112, 0.2205, 0.2130, 0.2170, 0.2217, 0.2153, 0.2200, 0.2188, 0.2310, 0.2310, 0.2340, 0.2433, 0.2404, 0.2433, 0.2491, 0.2509, 0.2532, 0.2497, 0.2503, 0.2515, 0.2570, 0.2590, 0.2625, 0.2620, 0.2596, 0.2631, 0.2655, 0.2695, 0.2678, 0.2649, 0.2655, 0.2701, 0.2690, 0.2719, 0.2666, 0.2713, 0.2754, 0.2830, 0.2789, 0.2748, 0.2789, 0.2806, 0.2800, 0.2800, 0.2841, 0.2818, 0.2783, 0.2847, 0.2859, 0.2859, 0.2870, 0.2835, 0.2841, 0.2830, 0.2812, 0.2824, 0.2870, 0.2853, 0.2882, 0.2853, 0.2923, 0.2911, 0.2882, 0.2929, 0.2940, 0.2958, 0.3016, 0.3022, 0.3005, 0.3034, 0.2092, 0.3110, 0.3110, 0.3127, 0.3186, 0.3191, 0.3151, 0.3133, 0.3256, 0.3162, 0.3238, 0.3256, 0.3285, 0.3238, 0.3232, 0.3285, 0.3285, 0.3256]

Tw_25_hidden_50_traccuracy = [0.1981, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2001, 0.2005, 0.2005, 0.2028, 0.2016, 0.2005, 0.1997, 0.2020, 0.2016, 0.2032, 0.2047, 0.2075, 0.2071, 0.2153, 0.2133, 0.2118, 0.2203, 0.2250, 0.2343, 0.2297, 0.2215, 0.2301, 0.2359, 0.2347, 0.2355, 0.2409, 0.2464, 0.2425, 0.2499, 0.2468, 0.2577, 0.2581, 0.2608, 0.2624, 0.2631, 0.2748, 0.2729, 0.2678, 0.2744, 0.2803, 0.2919, 0.2822, 0.2810, 0.2884, 0.2923, 0.2880, 0.2962, 0.2966, 0.2966, 0.3017, 0.3040, 0.2912, 0.3009, 0.3036, 0.3048, 0.3157, 0.3180, 0.3235, 0.3223, 0.3110, 0.3184, 0.3169, 0.3172, 0.3192, 0.3176, 0.3301, 0.3348, 0.3192, 0.3293, 0.3243, 0.3204, 0.3305, 0.3211, 0.3324, 0.3297, 0.3289, 0.3348, 0.3340, 0.3332, 0.3320, 0.3351, 0.3449, 0.3414, 0.3355]
Tw_25_hidden_50_valaccuracy = [0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814,0.1814, 0.1814, 0.1814, 0.1814, 0.2001, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1814, 0.1820, 0.1826, 0.1838, 0.1855, 0.1849, 0.1890, 0.1931, 0.1925, 0.1978, 0.2060, 0.2083, 0.2141, 0.2135, 0.2194, 0.2165, 0.2264, 0.2328, 0.2305, 0.2369, 0.2363, 0.2427, 0.2415, 0.2480, 0.2532, 0.2573, 0.2625, 0.2625, 0.2649, 0.2649, 0.2672, 0.2672, 0.2701, 0.2684, 0.2695, 0.2748, 0.2736, 0.2725, 0.2760, 0.2777, 0.2783, 0.2818, 0.2800, 0.2841, 0.2835, 0.2824, 0.2830, 0.2876, 0.2859, 0.2894, 0.2917, 0.2905, 0.2876, 0.2894, 0.2935, 0.2917, 0.2970, 0.2964,  0.2987, 0.2975, 0.2958, 0.3011, 0.2993, 0.2987, 0.2975, 0.2975,  0.2993, 0.3011, 0.3057, 0.3040, 0.3040, 0.3040, 0.3016, 0.3051, 0.3069, 0.3051, 0.3057, 0.3069, 0.3051]

Tw_25_hidden_75_traccuracy = [0.1604, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1911, 0.1907, 0.1915, 0.1911, 0.1954, 0.1954, 0.1931, 0.1966, 0.1997, 0.2036, 0.2086, 0.2024, 0.2047, 0.2110, 0.2102, 0.2075, 0.2114, 0.2083, 0.2180, 0.2067, 0.2083, 0.2227, 0.2199, 0.2301, 0.2199, 0.2293, 0.2242, 0.2336, 0.2316, 0.2336, 0.2386, 0.2371, 0.2343, 0.2433, 0.2429, 0.2472, 0.2495, 0.2398, 0.2487, 0.2441, 0.2534, 0.2503, 0.2530, 0.2546, 0.2573, 0.2589, 0.2612, 0.2659, 0.2639, 0.2600, 0.2663, 0.2659, 0.2768, 0.2748, 0.2772, 0.2826, 0.2768, 0.2803, 0.2830, 0.2807, 0.2861, 0.2865, 0.2768, 0.2869, 0.2838, 0.2896, 0.2826, 0.2896, 0.2873, 0.2779, 0.2888, 0.2896, 0.2939, 0.2966, 0.3005, 0.2892, 0.3083, 0.2943, 0.3040, 0.2970, 0.3044, 0.3083, 0.2954, 0.3060, 0.3052, 0.3005, 0.3063, 0.3040, 0.2986]
Tw_25_hidden_75_valaccuracy = [0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1949, 0.1954, 0.1954, 0.1954, 0.1966, 0.2036, 0.2083, 0.2077, 0.2124, 0.2159, 0.2194, 0.2200, 0.2200, 0.2194, 0.2194, 0.2258, 0.2322, 0.2299, 0.2427, 0.2421, 0.2450, 0.2415, 0.2433, 0.2439, 0.2456, 0.2474, 0.2450, 0.2474, 0.2520, 0.2509, 0.2544, 0.2538, 0.2532, 0.2555, 0.2497, 0.2579, 0.2596, 0.2520, 0.2520, 0.2590, 0.2625, 0.2602, 0.2643, 0.2631, 0.2625, 0.2596, 0.2649, 0.2643, 0.2695, 0.2771, 0.2730, 0.2754, 0.2730, 0.2806, 0.2754, 0.2841, 0.2865, 0.2865, 0.2865, 0.2888, 0.2975, 0.2946, 0.2929, 0.2923, 0.2987, 0.2981, 0.2970, 0.2993, 0.3016, 0.3057, 0.3063, 0.3069, 0.3081, 0.3086, 0.3098, 0.3040, 0.3040, 0.3168, 0.3127, 0.3086, 0.3116, 0.3116, 0.3051, 0.3116, 0.3098, 0.3081, 0.3092]

epochs = range(100)

plt.plot(Tw_25_hidden_25_valaccuracy)
plt.plot(Tw_25_hidden_50_valaccuracy)
plt.plot(Tw_25_hidden_75_traccuracy)