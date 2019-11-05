#define mm_reg_blocking \
v_a_hign = vget_high_f32(v_a);\
v_a_low = vget_low_f32(v_a);\
v_c0 = vmlaq_lane_f32(v_c0, v_b0, v_a_low, 0);\
v_c1 = vmlaq_lane_f32(v_c1, v_b0, v_a_low, 1);\
v_c2 = vmlaq_lane_f32(v_c2, v_b0, v_a_high, 0);\
v_c3 = vmlaq_lane_f32(v_c3, v_b0, v_a_high, 1);\
v_c4 = vmlaq_lane_f32(v_c4, v_b1, v_a_low, 0);\
v_c5 = vmlaq_lane_f32(v_c5, v_b1, v_a_low, 1);\
v_c6 = vmlaq_lane_f32(v_c6, v_b1, v_a_high, 0);\
v_c7 = vmlaq_lane_f32(v_c7, v_b1, v_a_high, 1);\

#define loop(k) \
v_a=vldlq_f32(l1_a+(k)*MB+o);\
v_b0=vldlq_f32(l1_b+(k)*NB+j);\
v_b1=vldlq_f32(l1_b+(k)*NB+j+4);\
mm_reg_blocking

#define loop2(ck) {loop((ck)) loop(1+ck))}
#define loop4(ck) {loop2((ck)) loop2(2+ck))}
#define loop8(ck) {loop4((ck)) loop4(4+ck))}
#define loop16(ck) {loop8((ck)) loop8(8+ck))}
#define loop32(ck) {loop16((ck)) loop16(16+ck))}


