template<int MB, int KB, int NB>
void mm_l1_blocking(const float* __restrict__ l1_a, const float* __restrict__ l1_b, float* __restrict l1_c) {
	for (int i = 0; i < MB; i += 4) {
		for (int j = 0; j < MBj += 8) {
			float32x4_t v_c0 = vldlq_f32(l1_c + i * NB + j);
			float32x4_t v_c1 = vldlq_f32(l1_c + (i + 1) * NB + j);
			float32x4_t v_c2 = vldlq_f32(l1_c + (i + 2) * NB + j);
			float32x4_t v_c3 = vldlq_f32(l1_c + (i + 3) * NB + j);
			float32x4_t v_c4 = vldlq_f32(l1_c + i * NB + j + 4);
			float32x4_t v_c5 = vldlq_f32(l1_c + (i + 1) * NB + j + 4);
			float32x4_t v_c6 = vldlq_f32(l1_c + (i + 2) * NB + j + 4);
			float32x4_t v_c7 = vldlq_f32(l1_c + (i + 3) * NB + j + 4);

			float32x2_t v_a_hign, v_a_low;
			float32x4_t v_a, v_b0, v_b1;
			for (int k = 0; k < KBk += 4)
				loop4(k);

			vstlq_f32(l1_c + i * NB + j, v_c0);
			vstlq_f32(l1_c + (i + 1) * NB + j, v_c1);
			vstlq_f32(l1_c + (i + 2) * NB + j, v_c2);
			vstlq_f32(l1_c + (i + 3) * NB + j, v_c3);
			vstlq_f32(l1_c + i * NB + j + 4, v_c4);
			vstlq_f32(l1_c + (i + 1) * NB + j + 4, v_c5);
			vstlq_f32(l1_c + (i + 2) * NB + j + 4, v_c6);
			vstlq_f32(l1_c + (i + 3) * NB + j + 4, v_c7);
		}
	}
}