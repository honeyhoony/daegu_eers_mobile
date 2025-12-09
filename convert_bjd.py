# convert_bjd.py (전체 교체)

# 1. 처리할 대구/경북 지역 코드 (앞 두 자리)
TARGET_REGIONS = ["27", "47"]

# 2. 결과 데이터를 저장할 파일명
OUTPUT_FILENAME = "bjd_mapper.py"

bjd_map = {}

print("법정동 코드 데이터 변환을 시작합니다 (동/읍/면 단위, '리' 제외)...")

try:
    with open("bjd_data.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 3:
                continue
            
            bjd_code, address, status = parts
            
            if "존재" in status:
                if bjd_code[:2] in TARGET_REGIONS:
                    # ▼▼▼ [최종 수정] '동'은 포함하고, '리'는 제외하는 조건으로 변경 ▼▼▼
                    is_dong_level = bjd_code[5:8] != "000"
                    
                    if is_dong_level and not address.endswith("리"):
                        code_key = bjd_code[:8]
                        full_address = " ".join(address.split())
                        
                        # 중복 코드 방지 (가장 구체적인 주소만 남김)
                        if code_key not in bjd_map:
                            bjd_map[code_key] = full_address
                            print(f"  - 추가: {code_key} -> {full_address}")
                    # ▲▲▲ [최종 수정] 완료 ▲▲▲

    print(f"\n총 {len(bjd_map)}개의 동/읍/면 코드를 추출했습니다.")

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        f.write("# bjd_mapper.py (자동 생성된 파일, 동/읍/면 단위, '리' 제외)\n\n")
        f.write("BJD_CODE_MAP = {\n")
        for code, name in sorted(bjd_map.items()):
            f.write(f'    "{code}": "{name}",\n')
        f.write("}\n\n")
        f.write("def get_bjd_name(code: str) -> str:\n")
        f.write('    """법정동 코드 10자리 중 앞 8자리를 사용해 동/읍/면 단위 주소를 찾습니다."""\n')
        f.write("    if not code or len(code) < 8:\n")
        f.write('        return ""\n')
        f.write("    return BJD_CODE_MAP.get(code[:8], \"\")\n")

    print(f"\n[성공] '{OUTPUT_FILENAME}' 파일이 성공적으로 생성되었습니다.")

except FileNotFoundError:
    print("\n[오류] 'bjd_data.txt' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"\n[오류] 파일 처리 중 문제가 발생했습니다: {e}")