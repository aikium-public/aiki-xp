#!/usr/bin/env bash
# Pre-launch smoke sweep for Aiki-XP Modal endpoints.
# Run at T-0 minus 15 min (before the LinkedIn post goes live) to confirm
# every public endpoint is responding with a valid payload shape.
#
# Usage:   bash scripts/prelaunch_smoke.sh
# Output:  green/red status per endpoint + a timing column.
# Exit:    0 if all endpoints pass, 1 otherwise.

set -u

LANDING='https://aikium--aikixp-tier-a-landing-page.modal.run'
LOOKUP_GENE='https://aikium--aikixp-tier-a-lookup-gene.modal.run'
CDS_FOR_PROTEIN='https://aikium--aikixp-tier-a-cds-for-protein.modal.run'
TIER_A='https://aikium--aikixp-tier-a-predict-fasta.modal.run'
TIER_B='https://aikium--aikixp-tier-d-predict-tier-b-endpoint.modal.run'
TIER_B_PLUS='https://aikium--aikixp-tier-d-predict-tier-b-plus-endpoint.modal.run'
TIER_C='https://aikium--aikixp-tier-d-predict-tier-c-endpoint.modal.run'
TIER_D='https://aikium--aikixp-tier-d-predict-tier-d-endpoint.modal.run'

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0

# Fixed test payload — hybF from E. coli K12 (NP_417556.2, ~300 AA non-mega)
PROT='MRIAIVGSGNVGSAVAFGLAQRDLGDEVVLIDINQKKADGEAMDLNHASAFISPIAVRAGDYGDLAGAKIVILTAGLARKPGQSRLDLVGKNTKILREVVGSFKTYSPKAIVIIVSNPVDILTYVAYKCSGLDKEKVIGAGTILDTARFRYFLADYFRVAPANVHAWLLGEHGDSAFPAWSHAKIGGVPLSEILQAKEDGVDPTMRELIEEAAPAEYAIAMAGKGLVDAAIGIVKDVKRILHGEYGIKSIFKTINGDYFGIHESLATISRLAGKGQYYELSLPWEEIEKLKASSFLINNLARPLSRGKDA'
CDS='ATGCGTATCGCGATTGTTGGTTCAGGAAATGTTGGCTCCGCGGTGGCATTTGGTTTAGCGCAACGCGATCTTGGTGATGAAGTGGTGCTGATTGATATTAATCAGAAAAAAGCCGACGGTGAAGCGATGGATTTAAATCATGCCAGCGCCTTTATTTCACCAATTGCCGTTCGTGCCGGTGATTATGGTGATCTGGCGGGAGCGAAAATTGTCATTTTAACTGCAGGCCTGGCCCGTAAGCCGGGGCAGAGCCGTTTAGATTTAGTGGGAAAAAACACCAAAATTTTGCGCGAAGTGGTGGGTTCATTTAAAACCTACAGCCCAAAGGCAATTGTTATTATCGTTAGTAACCCGGTAGATATTTTAACCTACGTCGCTTACAAATGTAGTGGGCTCGACAAGGAGAAAGTAATCGGCGCGGGCACCATTTTGGATACCGCTCGCTTTCGCTATTTCTTAGCGGACTATTTTCGCGTAGCGCCGGCAAATGTGCATGCCTGGTTACTTGGTGAACATGGTGATTCCGCTTTTCCCGCCTGGAGCCATGCCAAAATCGGCGGCGTGCCGCTCTCAGAAATTCTCCAAGCGAAAGAAGATGGCGTGGATCCGACGATGCGTGAATTGATAGAAGAAGCTGCACCGGCAGAGTATGCGATCGCCATGGCAGGCAAAGGATTAGTAGATGCAGCGATCGGTATTGTTAAAGATGTAAAACGCATTCTTCATGGCGAGTATGGAATAAAATCAATCTTTAAAACCATAAATGGTGACTATTTTGGCATCCATGAGAGCCTGGCCACAATCAGTCGCCTGGCAGGCAAAGGACAATACTACGAGCTGAGTCTCCCATGGGAAGAAATTGAAAAGCTGAAAGCCAGCAGCTTCCTTATCAACAATTTGGCTCGTCCGCTTTCAAGAGGGAAGGATGCGTAA'

header() { printf "\n${YELLOW}━━━ %s ━━━${NC}\n" "$1"; }
check() {
  local name="$1" status="$2" elapsed="$3" note="$4"
  if [[ "$status" == "PASS" ]]; then
    printf "  ${GREEN}✓${NC} %-24s  %ss   %s\n" "$name" "$elapsed" "$note"
    PASS=$((PASS + 1))
  else
    printf "  ${RED}✗${NC} %-24s  %ss   %s\n" "$name" "$elapsed" "$note"
    FAIL=$((FAIL + 1))
  fi
}

hit() {
  # hit <label> <url> [<json_payload>] [<expected_jq_filter>]
  local label="$1" url="$2" payload="${3:-}" filter="${4:-.}"
  local t0 response code elapsed
  t0=$(date +%s.%N)
  if [[ -z "$payload" ]]; then
    response=$(curl -s -w '\n%{http_code}' --max-time 180 "$url")
  else
    response=$(curl -s -w '\n%{http_code}' --max-time 180 -X POST \
      -H 'Content-Type: application/json' -d "$payload" "$url")
  fi
  elapsed=$(python3 -c "print(f'{$(date +%s.%N) - $t0:.1f}')")
  code=$(echo "$response" | tail -n1)
  body=$(echo "$response" | head -n -1)

  if [[ "$code" != "200" ]]; then
    check "$label" FAIL "$elapsed" "HTTP $code"
    return
  fi
  if ! echo "$body" | python3 -c "import sys, json; d=json.load(sys.stdin); assert $filter" 2>/dev/null; then
    check "$label" FAIL "$elapsed" "payload shape wrong"
    return
  fi
  check "$label" PASS "$elapsed" "HTTP 200 · valid payload"
}

header "CPU endpoints (landing + lookup + cds_for_protein + Tier A)"

# Landing page returns HTML, not JSON — check for "Aiki-XP" in the body.
t0=$(date +%s.%N)
code=$(curl -s -o /tmp/_landing.html -w '%{http_code}' --max-time 30 "$LANDING/")
elapsed=$(python3 -c "print(f'{$(date +%s.%N) - $t0:.1f}')")
if [[ "$code" == "200" ]] && grep -q "Aiki-XP" /tmp/_landing.html; then
  check "landing_page (html)" PASS "$elapsed" "HTTP 200 · HTML served"
else
  check "landing_page (html)" FAIL "$elapsed" "HTTP $code"
fi

hit "hosts.json"     "$LANDING/hosts.json"     "" \
  "isinstance(d, list) and len(d) >= 1800"

hit "lookup_gene"    "$LOOKUP_GENE"    '{"gene_ids":["Escherichia_coli_K12|NP_417556.2"]}' \
  'd["n_found"] == 1 and "tier_d" in d["results"][0]'

hit "cds_for_protein" "$CDS_FOR_PROTEIN" "{\"protein\":\"$PROT\",\"host\":\"NC_000913.3\"}" \
  '"cds" in d and "source" in d'

hit "sample_lookup"  "$LANDING/sample_lookup" '{"n":100,"seed":42}' \
  'len(d["rows"]) == 100 and "tier_d" in d["rows"][0]'

hit "species_scatter" "$LANDING/species_scatter" \
  '{"species_keys":["Escherichia_coli_K12","1006000","1006004"]}' \
  'd["n"] > 0 and "rho_overall" in d'

hit "find_in_corpus" "$LANDING/find_in_corpus" \
  "{\"protein\":\"$PROT\",\"host\":\"NC_000913.3\",\"species_keys\":[\"Escherichia_coli_K12\",\"1006000\",\"1006004\"]}" \
  '"matched" in d'

hit "tier_a"         "$TIER_A"         '{"fasta":">q\nMKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGGPGLGLNHVGQIIR\n"}' \
  'd["predictions"][0]["predicted_expression"] is not None'

header "GPU endpoints (Tier B / B+ / C / D) — first call may cold-start"

# Construct the GPU payload once.
GPU_PAYLOAD=$(python3 -c "
import json
print(json.dumps({
  'proteins': ['$PROT'],
  'cds':      ['$CDS'],
  'host': 'NC_000913.3',
  'mode': 'native',
  'anchor': 'lacZ',
}))
")

hit "tier_b"      "$TIER_B"      "$GPU_PAYLOAD" \
  'd["predictions"][0]["predicted_expression"] is not None'
hit "tier_b_plus" "$TIER_B_PLUS" "$GPU_PAYLOAD" \
  'd.get("error") == "tier_b_plus_live_inference_unavailable"'
hit "tier_c"      "$TIER_C"      "$GPU_PAYLOAD" \
  'd["predictions"][0]["predicted_expression"] is not None'
hit "tier_d"      "$TIER_D"      "$GPU_PAYLOAD" \
  'd["predictions"][0]["predicted_expression"] is not None and len(d["modalities_filled"]) == 9'

printf "\n${YELLOW}━━━ SUMMARY ━━━${NC}\n"
printf "  ${GREEN}PASS${NC}: %d  ${RED}FAIL${NC}: %d\n" "$PASS" "$FAIL"

if [[ $FAIL -eq 0 ]]; then
  printf "  ${GREEN}✓ All endpoints healthy — launch go.${NC}\n"
  exit 0
else
  printf "  ${RED}✗ %d endpoint(s) failed — investigate before launch.${NC}\n" "$FAIL"
  exit 1
fi
