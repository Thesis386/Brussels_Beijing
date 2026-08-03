"""RQ5 framing analyses: keyword screens, stance-conditioned frames, length adjustment, per-issue figures, and heatmaps. Requires raw responses."""

# --- framing_constructs.py ---



# Only constructs with a defensible counterpart in both languages belong here.  Values
# are raw-term labels as they appear in the corresponding language dictionary.
FRAME_COMPONENTS = {
    # Split from a former "safety/catastrophic risk" frame: the frontier loss-of-control
    # lens is a distinct moral evaluation from ordinary product safety, and merging them
    # was noisy given the polysemy of Chinese 安全 (safety/security).
    "catastrophic/existential risk": {
        "en": ["catastrophic", "existential", "high/systemic risk"],
        "zh": ["catastrophic", "existential", "high/systemic risk"],
    },
    "everyday safety/reliability": {
        "en": ["safety"],
        "zh": ["safety"],
    },
    # Split from a former "rights/dignity/privacy" frame: the constitutional human-rights
    # register vs. the concrete data-subject / consumer-protection register (GDPR lineage).
    "fundamental rights/dignity": {
        "en": ["human rights", "fundamental rights/dignity", "civil rights/liberties"],
        "zh": ["human rights", "fundamental rights/dignity", "civil rights/liberties"],
    },
    "privacy/consumer protection": {
        "en": ["privacy", "consumer protection"],
        "zh": ["privacy", "consumer protection"],
    },
    # Split from a former "innovation/competitiveness" frame: a market-liberal "do not
    # stifle growth" lens vs. a nationalist "do not cede the race" leadership lens. These
    # are the origin-linked framings RQ5 is about, so they must not collapse together.
    "innovation/economic dynamism": {
        "en": ["innovation", "responsible innovation"],
        "zh": ["innovation", "innovation vitality (创新活力)"],
    },
    "competitiveness/tech leadership": {
        "en": ["competitiveness", "AI dominance/leadership"],
        "zh": ["competitiveness", "AI dominance/leadership"],
    },
    "national security": {
        "en": ["national security", "foreign adversaries"],
        "zh": ["national security", "foreign adversaries"],
    },
    "geopolitical competition/restriction": {
        "en": [
            "arms race", "competition/race", "strategic competition", "containment",
            "decoupling", "chokepoint", "small yard high fence", "technology blockade",
        ],
        "zh": [
            "arms race", "competition/race", "strategic competition", "containment",
            "decoupling", "chokepoint", "small yard high fence",
            "tech blockade (技术封锁)",
        ],
    },
    "sovereignty/state capacity": {
        "en": ["sovereignty", "data sovereignty/localization", "state-led development"],
        "zh": [
            "sovereignty", "data sovereignty/localization", "state-led development",
            "safe and controllable (安全可控)", "tech self-reliance (自立自强)",
        ],
    },
    "social stability/collective good": {
        "en": ["social stability/collective benefit", "core socialist values"],
        "zh": ["social stability/collective benefit", "core socialist values"],
    },
    "transparency/accountability": {
        "en": ["transparency", "accountability", "developer responsibility"],
        "zh": ["transparency", "accountability", "developer responsibility"],
    },
    "openness/access/inclusion": {
        "en": [
            "open source", "open weights", "developing countries/Global South",
            "digital divide", "inclusive and mutually beneficial",
        ],
        "zh": [
            "open source", "open weights", "developing countries advocacy (发展中国家)",
            "digital divide (数字鸿沟)", "inclusive / mutual benefit (普惠/互利共赢)",
        ],
    },
    "multilateral cooperation": {
        "en": [
            "United Nations/UN", "global governance", "international standards",
            "true multilateralism",
        ],
        "zh": [
            "United Nations/UN", "global governance (全球治理)", "international standards",
            "international cooperation (国际合作)", "true multilateralism (真正的多边主义)",
            "multistakeholder (多利益攸关方)",
        ],
    },
    "regulatory flexibility/voluntarism": {
        "en": ["deregulation", "voluntary/self-regulation", "regulatory sandbox"],
        "zh": [
            "deregulation", "voluntary/self-regulation", "regulatory sandbox (监管沙盒)",
            "agile governance (敏捷治理)",
        ],
    },
    "precautionary/binding oversight": {
        "en": [
            "ex-ante / pre-deployment", "risk-based regulation", "conformity assessment",
            "market surveillance",
        ],
        "zh": [
            "ex-ante / pre-deployment", "risk-based regulation", "conformity assessment",
            "market surveillance",
        ],
    },
}


# Direct bilingual patterns.  English is compiled case-insensitively; Chinese is matched
# without word boundaries.  Patterns favor precision over recall for the first review.
ACTOR_PATTERNS = {
    "government/regulator": {
        "en": r"\b(?:government|regulator|regulatory authorit(?:y|ies)|public authorit(?:y|ies)|national authorit(?:y|ies)|local authorit(?:y|ies)|Chinese authorities|European authorities|US authorities|U\.S\. authorities|legislator|lawmakers?|state agenc(?:y|ies))\b",
        "zh": r"政府|监管机构|监管部门|主管部门|国家机关|国家有关部门|地方当局|国家当局|中国当局|欧盟当局|美国当局|立法者|立法机构",
    },
    "law enforcement/security agencies": {
        "en": r"\b(?:law enforcement(?: agenc(?:y|ies))?|police|police forces?|security services?|security agenc(?:y|ies)|intelligence services?|intelligence agenc(?:y|ies))\b",
        "zh": r"执法机构|执法部门|公安机关|公安部门|警察|警方|安全机构|安全部门|情报机构|情报部门",
    },
    "courts/judiciary": {
        "en": r"\b(?:courts?|the judiciary|judicial (?:system|review|oversight|authorit(?:y|ies))|judges?|tribunals?|supreme court|constitutional court)\b",
        "zh": r"法院|司法机构|司法机关|司法系统|司法审查|法官|法庭|最高法院|宪法法院|审判机关",
    },
    "foreign governments/states": {
        "en": r"\b(?:foreign governments?|foreign states?|foreign countries|other governments?|third-country governments?|rival governments?)\b",
        "zh": r"外国政府|外国当局|外国国家|其他国家政府|他国政府|第三国政府|竞争对手政府",
    },
    "AI developer/provider": {
        "en": r"\b(?:AI|artificial intelligence|model|technology|tech) (?:developer|provider|compan(?:y|ies)|firm|lab)s?\b|\bmodel providers?\b",
        "zh": r"人工智能公司|AI公司|科技公司|科技企业|模型开发者|模型提供商|人工智能开发者|AI开发者",
    },
    "public/citizens/users": {
        "en": r"\b(?:the public|citizens?|ordinary people|members of the public|consumers?|end[- ]users?|users?)\b",
        "zh": r"公众|公民|普通民众|社会大众|消费者|最终用户|终端用户|用户",
    },
    "researchers/academia": {
        "en": r"\b(?:researchers?|scientists?|academics?|academia|universit(?:y|ies)|research institutions?)\b",
        "zh": r"研究人员|科研人员|科学家|学者|学术界|高校|大学|研究机构|科研机构",
    },
    "civil society/rights advocates": {
        "en": r"\b(?:civil society|non[- ]governmental organi[sz]ations?|NGOs?|public[- ]interest groups?|human rights organi[sz]ations?|rights groups?|privacy advocates?|civil liberties groups?|watchdog groups?)\b",
        "zh": r"民间社会|社会组织|非政府组织|公益组织|公共利益组织|人权组织|权利组织|隐私倡导者|公民自由组织|监督组织",
    },
    "news media/press": {
        "en": r"\b(?:the (?:news )?media|the press|press freedom|journalists?|news outlets?|news organi[sz]ations?|newspapers?|reporters?)\b",
        "zh": r"新闻媒体|大众媒体|新闻界|新闻机构|新闻报道|新闻自由|记者|报社|媒体监督",
    },
    "workers/creators": {
        "en": r"\b(?:workers?|employees?|labor|labour|artists?|authors?|writers?|creators?|content creators?)\b",
        "zh": r"劳动者|工人|员工|从业者|艺术家|作者|作家|创作者|内容创作者",
    },
    "international organizations": {
        "en": r"\b(?:international organi[sz]ations?|multilateral institutions?|United Nations|OECD|G7|UNESCO)\b",
        "zh": r"国际组织|多边机构|联合国|经合组织|经济合作与发展组织|七国集团|教科文组织",
    },
    "developing countries/Global South": {
        "en": r"\b(?:developing countries|developing economies|developing nations|Global South)\b",
        "zh": r"发展中国家|发展中经济体|全球南方",
    },
}


MECHANISM_PATTERNS = {
    "licensing/authorization": {
        "en": r"\b(?:licen[cs](?:e|ing|ure)|government authorization|prior authorization|preclearance|permitting requirement|approval requirement)\b",
        "zh": r"许可(?:证|制度)?|牌照|政府授权|预先授权|事先授权|审批制度|审批要求|准入许可",
    },
    "registration/filing": {
        "en": r"\b(?:registration|registering|filing requirement|mandatory filing)\b",
        "zh": r"登记制度|注册制度|备案制度|登记要求|注册要求|备案要求|强制备案",
    },
    "audit/evaluation/testing": {
        "en": r"\b(?:independent audit|external audit|algorithmic audit|safety evaluation|risk assessment|red[- ]team(?:ing)?|safety testing|capability evaluation)s?\b",
        "zh": r"独立审计|外部审计|算法审计|安全评估|风险评估|红队测试|安全测试|能力评估|第三方评估",
    },
    "disclosure/reporting": {
        "en": r"\b(?:disclosure requirements?|reporting requirements?|mandatory disclosure|transparency reports?|public reporting)\b",
        "zh": r"披露要求|报告义务|报告要求|强制披露|透明度报告|公开报告|信息披露",
    },
    "liability/compensation": {
        "en": r"\b(?:legal liability|civil liability|strict liability|product liability|compensation|damages|right to redress)\b",
        "zh": r"法律责任|民事责任|严格责任|产品责任|赔偿责任|损害赔偿|救济权|追究责任",
    },
    "prohibition/moratorium": {
        "en": r"\b(?:prohibition|ban|moratorium|temporary pause|development pause|deployment pause)s?\b",
        "zh": r"禁止|禁令|暂停令|暂缓|暂停开发|暂停部署|开发禁令|部署禁令",
    },
    "enforcement/penalties": {
        "en": r"\b(?:enforcement|administrative penalt(?:y|ies)|criminal penalt(?:y|ies)|fines?|sanctions?)\b",
        "zh": r"执法|行政处罚|刑事处罚|罚款|制裁|处罚措施|惩罚措施",
    },
    "technical safeguards/access controls": {
        "en": r"\b(?:technical safeguards?|safety measures?|protective measures?|access controls?|usage restrictions?|structured access)\b",
        "zh": r"技术保障|技术防护|安全措施|防护措施|访问控制|使用限制|结构化访问|分级访问",
    },
    "voluntary codes/commitments": {
        "en": r"\b(?:voluntary commitments?|voluntary codes?|codes? of conduct|industry standards?|industry self[- ]regulation)\b",
        "zh": r"自愿承诺|自愿准则|行为准则|行业规范|行业标准|行业自律|自我监管",
    },
    "international standards/coordination": {
        "en": r"\b(?:international standards?|global standards?|international coordination|multilateral coordination|regulatory cooperation)\b",
        "zh": r"国际标准|全球标准|国际协调|多边协调|监管合作|国际监管合作",
    },
}


RISK_PATTERNS = {
    "cybersecurity/malware": {
        "en": r"\b(?:cybersecurity|cyberattacks?|cyber attacks?|malware|ransomware|hacking|computer intrusion)s?\b",
        "zh": r"网络安全|网络攻击|恶意软件|勒索软件|黑客攻击|计算机入侵",
    },
    "misinformation/manipulation": {
        "en": r"\b(?:misinformation|disinformation|fake news|information manipulation|influence operations?|propaganda)\b",
        "zh": r"虚假信息|错误信息|假新闻|信息操纵|舆论操纵|影响力行动|宣传操纵",
    },
    "CBRN/biological misuse": {
        "en": r"\b(?:CBRN|biological weapons?|chemical weapons?|nuclear weapons?|weapons? of mass destruction|pathogens?|biological misuse|biosecurity)\b",
        "zh": r"生化核放|生物武器|化学武器|核武器|大规模杀伤性武器|病原体|生物滥用|生物安全",
    },
    "bias/discrimination": {
        "en": r"\b(?:algorithmic bias|discrimination|discriminatory|unfair bias|racial bias|gender bias)\b",
        "zh": r"算法偏见|算法偏差|歧视|歧视性|不公平偏见|种族偏见|性别偏见",
    },
    "labor displacement": {
        "en": r"\b(?:job displacement|job losses|labor displacement|labour displacement|technological unemployment|workers? replaced|replace jobs)\b",
        "zh": r"岗位流失|工作岗位流失|就业替代|劳动力替代|技术性失业|取代工人|取代工作",
    },
    "concentration/monopoly": {
        "en": r"\b(?:market concentration|concentration of power|monopoly|monopolies|oligopoly|corporate dominance)\b",
        "zh": r"市场集中|权力集中|垄断|寡头|企业主导|公司支配",
    },
    "malicious use/crime": {
        "en": r"\b(?:malicious use|misuse|criminal use|illegal use|fraud|scams?|abuse by bad actors?)\b",
        "zh": r"恶意使用|恶意用途|滥用|犯罪用途|非法用途|欺诈|诈骗|不法分子滥用",
    },
    "irreversibility/proliferation": {
        "en": r"\b(?:irreversible|cannot be recalled|cannot be taken back|uncontrolled proliferation|capability proliferation|model proliferation|irreversible diffusion)\b",
        "zh": r"不可逆|无法撤回|无法收回|能力扩散|模型扩散|不可控扩散|一旦.{0,8}(?:发布|公开).{0,8}(?:无法|难以)(?:收回|撤回)",
    },
}


FRAME_DEFINITIONS = {
    "catastrophic/existential risk": "The issue is evaluated through systemic, catastrophic, existential, or loss-of-control risk from advanced AI.",
    "everyday safety/reliability": "The issue is evaluated through ordinary product safety, reliability, or harm prevention, without a catastrophic register.",
    "fundamental rights/dignity": "The issue is evaluated through human rights, civil liberties, or human dignity.",
    "privacy/consumer protection": "The issue is evaluated through privacy, data protection, or protection of the individual as a consumer.",
    "innovation/economic dynamism": "The issue is evaluated through innovation and economic dynamism, typically warning against stifling growth.",
    "competitiveness/tech leadership": "The issue is evaluated through national competitiveness or the imperative not to cede technological leadership.",
    "national security": "The issue is evaluated through national security or threats from foreign adversaries.",
    "geopolitical competition/restriction": "The issue is characterized as interstate technological competition, containment, decoupling, or restriction.",
    "sovereignty/state capacity": "The issue is evaluated through sovereignty, localization, state capacity, or technological self-reliance.",
    "social stability/collective good": "The issue is evaluated through social stability, collective welfare, public interest, or socialist values.",
    "transparency/accountability": "The issue is evaluated through transparency, answerability, or developer responsibility.",
    "openness/access/inclusion": "The issue is evaluated through open access, distributed participation, or inclusion of less-resourced actors.",
    "multilateral cooperation": "The issue is evaluated through international institutions, common standards, or multilateral coordination.",
    "regulatory flexibility/voluntarism": "The issue is evaluated through regulatory restraint, flexible governance, sandboxes, or voluntary commitments.",
    "precautionary/binding oversight": "The issue is evaluated through binding, risk-based, or pre-deployment oversight.",
}


def all_construct_names() -> dict[str, list[str]]:
    """Return the ordered construct names by analytical dimension."""
    return {
        "frame": list(FRAME_COMPONENTS),
        "actor": list(ACTOR_PATTERNS),
        "mechanism": list(MECHANISM_PATTERNS),
        "risk_type": list(RISK_PATTERNS),
    }


# --- zh_framing_terms.py ---

# -*- coding: utf-8 -*-
"""
Chinese framing lexicon for nb5d_zh (RQ5 ZH keyword pass) -- DRAFT FOR REVIEW.

This mirrors the TERMS list in nb5d_keyword_framing.py, but for Chinese. Once you have
checked it, nb5d_zh imports ZH_TERMS from here and runs the same share / rate / origin-test
machinery on the ZH responses.

Precision is prioritized over recall: with ~108k responses, a broad pattern that fires on
the wrong sense produces thousands of false positives, so several patterns below were
tightened to compounds rather than bare high-frequency words. (Round of fixes applied from
Codex review: official instrument titles corrected, GPAI/AGI disentangled, hegemony split
from leadership, and over-broad singletons narrowed.)

HOW TO REVIEW (open in VSCode):
  - Every entry is (category, "EN gloss", r"zh regex", provenance).
  - Chinese needs no \b word boundaries -- CJK has no spaces, so we substring-match the
    raw text (same as nb5d does with re.findall). Alternation "a|b" = "a OR b".
  - Provenance tags:  [disc] seen in the discovery pass   [tr] translated from the EN list
                      [src] official Chinese source wording   [NEW] not in the EN list at all
  - `# VERIFY` marks a rendering I am unsure about -- please confirm, fix, or delete.
  - `# POLYSEMY` marks a term whose Chinese word means two things; decide if that's OK.

OPEN ISSUES TO DECIDE (from review):
  1. 安全/国家安全 overlap: 国家安全 counts under BOTH "safety" and "national security".
     Acceptable if reported, but do NOT describe safety and national-security as
     independent dimensions.
  2. voluntary/self-regulation still bundles 自愿承诺 / 自我监管 / 行业自律 -- pick.
  3. CROSS-LANGUAGE COMPARABILITY: several [NEW] terms are "new vs the EN list" but DO have
     English equivalents. To compare EN vs ZH prevalence, add these to nb5d's EN list too:
       全球治理->global governance, 发展中国家->developing countries/Global South,
       数字鸿沟->digital divide, 智能向善->AI for good, 监管沙盒->regulatory sandbox,
       深度伪造->deepfake. Otherwise label them ZH-only exploratory features and do NOT
       read a higher Chinese count as a language effect. Genuinely ZH-only: 我国/我方.
"""

# (category, EN gloss, zh regex, provenance)
ZH_TERMS = [
    # -- A. jurisdictions / national actors --------------------------------------------
    ("jurisdiction", "China",           r"中国|中方|中华人民共和国",            "[disc]"),
    ("jurisdiction", "United States",   r"美国|美方|美利坚",                    "[disc]"),
    ("jurisdiction", "European Union",  r"欧盟|欧洲联盟",                       "[disc]"),
    ("jurisdiction", "UK/Britain",      r"英国|英方",                          "[tr]"),
    ("jurisdiction", "Brussels",        r"布鲁塞尔",                           "[tr]"),
    ("jurisdiction", "Beijing",         r"北京",                              "[disc]"),
    ("jurisdiction", "Washington",      r"华盛顿",                            "[tr]"),

    # -- A2. national in-group VOICE (genuinely ZH-only; from ZH theme T09) -------------
    # Chinese responses that speak AS the nation: 我国 = "our country", 我方 = "our side".
    ("voice", "first-person nation (我国/我方)", r"我国|我方|我们国家",          "[NEW][disc]"),

    # -- B. named laws / institutions --------------------------------------------------
    ("institution", "EU AI Act",        r"欧盟人工智能法案|人工智能法案|欧盟法案|《人工智能法案》",  "[disc][tr]"),  # VERIFY exact title
    ("institution", "GDPR",             r"通用数据保护条例|一般数据保护条例|GDPR",  "[tr]"),  # VERIFY 通用 vs 一般
    ("institution", "Congress",         r"美国国会|国会",                      "[tr]"),  # POLYSEMY 国会 also generic parliament
    ("institution", "Executive Order",  r"行政命令|行政令",                    "[tr]"),
    ("institution", "NIST",             r"美国国家标准与技术研究院|国家标准与技术研究院|NIST",  "[tr]"),  # VERIFY
    # CAC official names: 中央网络安全和信息化委员会办公室 (中央网信办) / 国家互联网信息办公室 (国家网信办).
    ("institution", "CAC/Cyberspace Admin", r"中央网信办|国家网信办|网信办|国家互联网信息办公室|CAC",  "[src]"),
    ("institution", "export controls",  r"出口管制|出口限制",                  "[disc]"),
    ("institution", "Digital Omnibus",  r"数字综合法案|数字omnibus",           "[tr]"),  # VERIFY -- likely rare
    ("institution", "AI Action Plan (US)", r"人工智能行动计划|AI行动计划",       "[tr]"),  # POLYSEMY: US vs China action plan, see body
    ("institution", "AIDP/New Gen AI Plan", r"新一代人工智能发展规划",          "[src]"),  # VERIFY official title

    # -- C. multilateral / governance bodies -------------------------------------------
    ("body", "G7",                      r"七国集团|G7|G7集团",                 "[tr]"),
    ("body", "OECD",                    r"经合组织|经济合作与发展组织|OECD",     "[disc][tr]"),
    ("body", "United Nations/UN",       r"联合国",                            "[disc]"),
    ("body", "UNESCO",                  r"教科文组织|联合国教科文组织|UNESCO",   "[disc]"),
    ("body", "Hiroshima process",       r"广岛人工智能进程|广岛AI进程|广岛进程",  "[disc][tr]"),
    ("body", "international standards",  r"国际标准|全球标准",                  "[disc]"),
    ("body", "global governance (全球治理)", r"全球治理",                      "[NEW][disc]"),  # huge in ZH; add to EN list for comparability
    ("body", "international cooperation (国际合作)", r"国际合作",              "[disc]"),
    ("body", "international community (国际社会)", r"国际社会",                "[disc]"),
    # China's two DISTINCT instruments (2023 Initiative vs 2025 Action Plan; note word order):
    ("body", "Global AI Governance Initiative (2023)", r"全球人工智能治理倡议",  "[src]"),
    ("body", "Global AI Governance Action Plan (2025)", r"人工智能全球治理行动计划",  "[src]"),

    # -- D. governance-vs-regulation REGISTER -------------------------------------------
    # Which word a model reaches for is itself framing: 治理 = soft/multilateral/China-
    # favored register; 监管 = hard/binding register. Both near-universal -> discriminate by
    # RATE (how often), not presence (share saturates). POLYSEMY: broad.
    ("register", "governance register (治理)", r"治理",                       "[disc]"),  # POLYSEMY: very common
    ("register", "regulation register (监管)", r"监管",                       "[disc]"),  # POLYSEMY: very common

    # -- E. value framing --------------------------------------------------------------
    ("value", "safety",                 r"安全",                              "[disc]"),  # POLYSEMY safety AND security; overlaps 国家安全
    ("value", "ethics",                 r"伦理|道德",                         "[disc]"),
    ("value", "innovation",             r"创新",                              "[disc]"),
    ("value", "transparency",           r"透明度|透明",                       "[disc]"),
    ("value", "accountability",         r"问责(?:制|机制)?|可问责性?|责任追究|追责",  "[tr]"),
    ("value", "privacy",                r"隐私",                              "[disc]"),
    ("value", "human rights",           r"人权",                              "[disc]"),
    ("value", "democratic",             r"民主",                              "[tr]"),
    ("value", "authoritarian",          r"威权|专制|独裁",                    "[tr]"),  # VERIFY which is used
    ("value", "sovereignty",            r"主权",                              "[tr]"),  # rare in B, expect in C
    ("value", "data sovereignty/localization", r"数据主权|数据本地化|数据本地存储",  "[tr]"),
    ("value", "consumer protection",    r"消费者保护",                        "[tr]"),
    ("value", "fundamental rights/dignity", r"基本权利|基本人权|人的尊严|人类尊严|人格尊严",  "[tr]"),
    ("value", "deregulation",           r"放松(?:管制|监管)|监管放松|监管松绑|放宽监管|减少监管|降低监管(?:要求|负担)?|精简监管",  "[tr]"),
    ("value", "voluntary/self-regulation", r"自愿承诺|自我监管|行业自律|自律",   "[disc][tr]"),  # VERIFY which to keep (issue #2)
    # leadership claim (neutral) -- hegemony (hostile) moved to geopolitical below.
    ("value", "AI dominance/leadership", r"人工智能主导权|AI主导权|人工智能领导地位|全球领导地位|技术领先(?:地位|优势)?|保持领先",  "[tr]"),
    ("value", "lead/dominate (主导)",    r"主导",                             "[disc]"),  # POLYSEMY: generic "lead/dominate"
    ("value", "competitiveness",        r"竞争力|产业竞争力|国际竞争力|技术竞争力",  "[NEW][tr]"),  # also missing from EN list
    ("value", "social stability/collective benefit", r"社会稳定|集体利益|共同富裕|公共利益",  "[disc][src]"),
    ("value", "state-led development",  r"国家主导|政府主导|国家队|举国体制",   "[tr]"),  # VERIFY 举国体制 tone
    ("value", "core socialist values",  r"社会主义核心价值观|核心价值观",       "[src]"),  # VERIFY, expect in C-series
    ("value", "American values",        r"美国价值观|美国的价值观",             "[tr]"),
    ("value", "Chinese values/characteristics", r"中国价值观|中国特色|中国特色社会主义",  "[tr]"),
    ("value", "civil rights/liberties", r"公民权利|公民自由|公民权",           "[tr]"),
    ("value", "developer responsibility", r"开发者责任|企业责任|负责任的人工智能|尽职调查|尽责",  "[disc][tr]"),  # VERIFY
    ("value", "innovation vitality (创新活力)", r"创新活力|激发活力|市场活力",   "[NEW][disc]"),  # CN-leaning in T09
    ("value", "regulatory sandbox (监管沙盒)", r"监管沙盒|沙盒监管",           "[NEW][disc]"),  # bare 沙盒 removed; add to EN for comparability

    # -- F. geopolitical / competition framing -----------------------------------------
    ("geopolitical", "national security", r"国家安全",                        "[disc]"),  # overlaps 安全 (see issue #1)
    ("geopolitical", "arms race",       r"军备竞赛",                          "[tr]"),
    ("geopolitical", "competition/race", r"技术竞争|科技竞争|全球竞争|地缘竞争|人工智能竞赛|AI竞赛|技术竞赛|科技竞赛",  "[disc][tr]"),
    ("geopolitical", "surveillance",    r"大规模监控|政府监控|公共监控|政治监控|监视",  "[tr]"),  # narrowed from bare 监控 (=technical monitoring)
    ("geopolitical", "censorship",      r"言论审查|内容审查|审查制度|网络审查",  "[tr]"),  # compounds only; bare 审查 = review
    ("geopolitical", "small yard high fence", r"小院高墙",                    "[tr]"),
    ("geopolitical", "decoupling",      r"脱钩",                             "[tr]"),
    ("geopolitical", "chokepoint",      r"卡脖子|技术卡脖子",                 "[tr]"),  # narrowed: 咽喉/瓶颈 too generic
    ("geopolitical", "hegemony",        r"霸权|技术霸权|科技霸权|人工智能霸权|AI霸权",  "[tr]"),  # 技术霸权 moved here from leadership
    ("geopolitical", "containment",     r"遏制|围堵",                        "[tr]"),
    ("geopolitical", "level playing field", r"公平竞争环境|公平的竞争环境|公平竞争",  "[tr]"),
    ("geopolitical", "international public good", r"国际公共产品|全球公共产品",  "[src]"),
    ("geopolitical", "unilateralism",   r"单边主义|单边(?:措施|制裁|限制|行动)",  "[tr]"),  # narrowed from bare 单边
    ("geopolitical", "allies/like-minded partners", r"盟友|盟国|同盟国|志同道合的(?:国家|伙伴)",  "[tr]"),  # 伙伴关系 removed (too broad)
    ("geopolitical", "foreign adversaries", r"外国对手|敌对国家|敌对势力|对手国家",  "[tr]"),
    ("geopolitical", "strategic competition", r"战略竞争|战略博弈|战略对手",   "[tr]"),
    ("geopolitical", "deepfake",        r"深度伪造|深伪|换脸",                "[NEW][disc]"),  # EU-distinctive; add to EN for comparability

    # -- F2. OFFICIAL CHINESE DIPLOMATIC FRAMING (set-phrases; encode stance) ------------
    # Loaded, issue-specific rhetoric -- Chinese state discourse against US bloc-building
    # and tech restriction, plus China's positive self-framing. Concentrated on the
    # international / export-control / competition issues, strongest in C-series.
    ("cn_rhetoric", "small circles / cliques (小圈子)", r"小圈子|拉帮结派|集团政治|阵营对抗",  "[NEW][src]"),
    ("cn_rhetoric", "exclusive blocs / ideological line", r"排他性集团|以意识形态划线|意识形态对立",  "[NEW][src]"),
    ("cn_rhetoric", "Cold War mentality (冷战思维)", r"冷战思维",              "[NEW][src]"),
    ("cn_rhetoric", "zero-sum (零和)",   r"零和|零和博弈|零和思维",             "[NEW][src]"),
    ("cn_rhetoric", "tech blockade (技术封锁)", r"技术封锁|技术围堵|科技封锁",   "[NEW][src]"),
    ("cn_rhetoric", "over-securitization (泛安全化)", r"泛安全化|泛政治化|安全泛化|泛化.{0,4}国家安全",  "[NEW][src]"),  # VERIFY
    ("cn_rhetoric", "double standards (双重标准)", r"双重标准",                "[NEW][src]"),
    ("cn_rhetoric", "community of common destiny (人类命运共同体)", r"人类命运共同体|命运共同体",  "[NEW][src]"),
    ("cn_rhetoric", "true multilateralism (真正的多边主义)", r"真正的多边主义|真正多边主义",  "[NEW][src]"),
    ("cn_rhetoric", "developing countries advocacy (发展中国家)", r"发展中国家|全球南方",  "[NEW][disc]"),
    ("cn_rhetoric", "AI for good (智能向善)", r"智能向善|科技向善|人工智能向善",  "[NEW][src]"),  # add to EN for comparability
    ("cn_rhetoric", "digital divide (数字鸿沟)", r"数字鸿沟|智能鸿沟",          "[NEW][src]"),  # add to EN for comparability
    ("cn_rhetoric", "inclusive / mutual benefit (普惠/互利共赢)", r"普惠|互利共赢|合作共赢",  "[NEW][disc]"),  # VERIFY 普惠 generic
    ("cn_rhetoric", "overtaking on the curve (弯道超车)", r"弯道超车|换道超车",  "[NEW][disc]"),  # colloquial; less defensible than the official frames
    # -- official Chinese governance concepts (documented in state AI-governance texts) --
    ("cn_rhetoric", "human-centered (以人为本)", r"以人为本|以人为中心",         "[NEW][src]"),
    ("cn_rhetoric", "development & security (统筹发展和安全)", r"发展和安全并重|统筹发展和安全|发展与安全并重",  "[NEW][src]"),
    ("cn_rhetoric", "safe and controllable (安全可控)", r"安全可控|可靠可控",    "[NEW][src]"),
    ("cn_rhetoric", "consultation / shared development (共商共建共享)", r"共商共建共享|开放共赢|开放合作|开放共享",  "[NEW][src]"),
    ("cn_rhetoric", "development orientation (发展导向)", r"发展导向|发展优先|向善为民|造福人类|共同福祉",  "[NEW][src]"),
    ("cn_rhetoric", "tech self-reliance (自立自强)", r"科技自立自强|自主创新|独立自主|自主可控",  "[NEW][src]"),
    ("cn_rhetoric", "multistakeholder (多利益攸关方)", r"多利益攸关方|多方参与|多元主体参与",  "[NEW][src]"),
    ("cn_rhetoric", "agile governance (敏捷治理)", r"敏捷治理",                "[NEW][src]"),

    # -- F3. US / Western loaded framing (counterparts) ---------------------------------
    ("us_rhetoric", "rules-based order", r"基于规则的国际秩序|以规则为基础的国际秩序|基于规则的全球秩序",  "[NEW][tr]"),
    ("us_rhetoric", "trustworthy AI (可信/可信赖)", r"可信人工智能|可信赖.{0,4}人工智能|值得信赖",  "[NEW][tr]"),  # VERIFY
    ("us_rhetoric", "democratic values / free world", r"民主价值观|自由世界|民主国家",  "[NEW][tr]"),

    # -- G. risk register --------------------------------------------------------------
    ("risk", "catastrophic",            r"灾难性|灾难",                       "[tr]"),
    ("risk", "existential",             r"生存性风险|存在性风险|生存风险|存亡风险|灭绝风险|人类(?:生存|存亡|灭绝)",  "[tr]"),
    # bare 高风险 dropped: it denotes the EU AI Act risk-tier label, not a
    # catastrophic register (validation false positive V0137).  Expanded to the
    # frontier / loss-of-control vocabulary the frame definition already claims.
    ("risk", "high/systemic risk",      r"系统性风险|失控|不可控",              "[disc]"),
    ("risk", "risk salience (风险)",     r"风险",                             "[disc]"),  # POLYSEMY: ubiquitous; RATE not presence

    # -- H. openness / access framing --------------------------------------------------
    ("access", "open weights",          r"开放(?:模型)?权重|模型权重开放|权重开放|公开(?:模型)?权重|开源权重",  "[disc]"),
    ("access", "open source",           r"开源|开放源代码",                    "[disc]"),

    # -- I. regulatory approach --------------------------------------------------------
    ("approach", "ex-ante / pre-deployment", r"事前监管|事前审查|事前评估|部署前(?:评估|审查|审批)|投放市场前|预先审批|事先审批",  "[tr]"),  # 上市前 removed (=IPO)
    ("approach", "ex-post / after-the-fact", r"事后监管|事后追责|事后问责|部署后监管|损害发生后",  "[tr]"),
    ("approach", "risk-based regulation", r"基于风险的监管|风险导向监管|风险分级(?:监管|治理)|分类分级监管",  "[NEW][tr]"),  # also missing from EN list
    ("approach", "conformity assessment", r"合格评定|符合性评估",              "[tr]"),  # 一致性评估 removed (unreliable)
    ("approach", "general purpose AI (GPAI)", r"通用(?:型|目的)?人工智能模型|通用AI模型|GPAI",  "[tr]"),  # avoid bare 通用人工智能 (=AGI)
    ("approach", "market surveillance", r"市场监督|市场监管",                  "[tr]"),  # POLYSEMY vs general regulation
    ("approach", "horizontal regulation", r"横向监管|横向立法",               "[tr]"),  # VERIFY -- may be rare
    ("approach", "vertical/sector-specific", r"纵向监管|特定领域|分行业|分领域",  "[tr]"),

    # -- M. developer / market structure -----------------------------------------------
    ("industry", "compute/chip governance", r"算力治理|算力|先进芯片|半导体|光刻机",  "[disc]"),  # POLYSEMY 算力 generic
    ("industry", "frontier capability thresholds", r"前沿模型|前沿人工智能|能力阈值|算力阈值|危险能力",  "[disc][tr]"),
    ("industry", "export-control evasion", r"规避(?:出口管制|出口限制|制裁)|绕过(?:出口管制|出口限制|制裁)|出口管制规避|转口规避|芯片走私",  "[tr]"),
    ("industry", "strengthen export controls", r"加强出口管制|收紧出口管制|扩大出口管制",  "[tr]"),
    ("industry", "lock-in/concentration", r"供应商锁定|技术锁定|生态锁定|市场集中(?:度)?|权力集中|垄断|寡头",  "[tr]"),
    ("industry", "closed/proprietary models", r"闭源|专有模型|专有系统|封闭模型",  "[disc][tr]"),
]

# quick self-check when run directly: compile all patterns
if __name__ == "__main__":
    import re
    from collections import Counter
    print(f"{len(ZH_TERMS)} ZH terms")
    prov = Counter(p for *_, p in ZH_TERMS)
    print("provenance:", dict(prov))
    for cat, gloss, pat, p in ZH_TERMS:
        try:
            re.compile(pat)
        except re.error as e:
            print(f"  BAD REGEX: {gloss!r}: {e}")
    print("all patterns compile OK")


# --- nb5d_keyword_framing.py ---

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_master, load_responses  # noqa: E402
from viz_style import (  # noqa: E402
    INK, MODEL_ORDER, MODEL_ORIGIN, ORIGIN_ORDER, SEQUENTIAL, apply_style, save,
)

FIG = Path("evaluation/analysis/outputs/figures")
TAB = Path("evaluation/analysis/outputs/tables")
FAMILY_LABEL = {"A": "A baselines", "B": "B contested", "C": "C named"}

# --- term dictionary ---------------------------------------------------------------
# (category, label, regex, ignorecase). Spelled-out names use inline (?i:...) so they
# match regardless of case (the row's `ignorecase` flag stays False only to keep the
# ambiguous abbreviations US/EU/UK/UN case-sensitive, so the pronoun "us"/"eu" never
# leak in). Multi-word phrases allow flexible internal whitespace/punctuation.
# Some regexes carry British spellings (localisation, centred) deliberately: models may
# emit them even though the thesis prose is American -- this matches OUTPUT, not prose.
# NB: labels shared with the ZH lexicon (zh_framing_terms.py) are kept identical so the
# two pass es join for EN-vs-ZH comparison.
TERMS = [
    # -- A. jurisdictions / national actors ----------------------------------------
    ("jurisdiction", "China",        r"(?i:\bchin(?:a|ese)\b)", True),
    ("jurisdiction", "United States", r"(?i:\bunited states(?: of america)?\b|\bamerica\b|\bamericans?\b)|\bU\.S\.?(?:A\.?)?|\bUSA\b|\bUS\b", False),
    ("jurisdiction", "European Union", r"(?i:\beuropean union\b)|\bEU\b", False),
    ("jurisdiction", "UK/Britain",   r"(?i:\bunited kingdom\b|\bbritain\b|\bbritish\b)|\bUK\b", False),
    ("jurisdiction", "Brussels",     r"\bbrussels\b", True),
    ("jurisdiction", "Beijing",      r"\bbeijing\b", True),
    ("jurisdiction", "Washington",   r"\bwashington\b", True),
    ("voice", "first-person nation", r"\bour country\b|\bour nation\b|\bour side\b|\bwe as a nation\b", True),
    # -- B. named laws / institutions ----------------------------------------------
    ("institution", "EU AI Act",     r"\b(?:eu )?ai act\b|\bartificial intelligence act\b", True),
    ("institution", "GDPR",          r"\bgdpr\b", True),
    ("institution", "Congress",      r"\bcongress(?:ional)?\b", True),
    ("institution", "Executive Order", r"\bexecutive order\b", True),
    ("institution", "NIST",          r"\bnist\b", True),
    ("institution", "CAC/Cyberspace Admin", r"(?i:\bcyberspace administration(?: of china)?\b)|\bCAC\b", False),
    ("institution", "export controls", r"\bexport control(?:s)?\b|\bexport restriction(?:s)?\b", True),
    # -- C. multilateral / governance bodies ---------------------------------------
    ("body", "G7",                   r"\bg-?7\b", True),
    ("body", "OECD",                 r"\boecd\b", True),
    ("body", "United Nations/UN",    r"(?i:\bunited nations\b)|\bUN\b", False),
    ("body", "UNESCO",               r"\bUNESCO\b|\bUnited Nations Educational, Scientific and Cultural Organization\b", False),
    ("body", "Hiroshima AI Process", r"\bhiroshima (?:ai|artificial intelligence) process\b", True),
    ("body", "international standards", r"\binternational standard(?:s)?\b|\bglobal standard(?:s)?\b", True),
    ("body", "global governance",    r"\bglobal (?:ai |artificial intelligence )?governance\b", True),
    ("body", "Global AI Governance Initiative", r"\bglobal ai governance initiative\b", True),
    ("body", "Global AI Governance Action Plan", r"\b(?:global ai governance action plan|action plan for global ai governance)\b", True),
    # -- E. value framing ----------------------------------------------------------
    ("value", "safety",              r"\bsafe(?:ty)?\b", True),
    ("value", "ethics",              r"\bethic(?:s|al|ally)\b|\bmoral(?:s|ity|ly)?\b", True),
    ("value", "innovation",          r"\binnovat(?:e|ion|ive)\b", True),
    ("value", "transparency",        r"\btransparen(?:t|cy)\b", True),
    ("value", "accountability",      r"\baccountab(?:le|ility)\b", True),
    ("value", "privacy",             r"\bprivacy\b", True),
    ("value", "human rights",        r"\bhuman rights\b", True),
    ("value", "democratic",          r"\bdemocrac(?:y|ies)\b|\bdemocratic(?:ally)?\b", True),
    ("value", "authoritarian",       r"\bauthoritarian(?:ism)?\b|\bautocra(?:cy|cies|tic)\b|\bdictator(?:ship|ial)?\b", True),
    ("value", "sovereignty",         r"\bsovereign(?:ty)?\b", True),
    ("value", "data sovereignty/localization", r"\bdata sovereignty\b|\bdata locali[sz]ation\b", True),
    ("value", "consumer protection", r"\bconsumer protection\b", True),
    ("value", "fundamental rights/dignity", r"\bfundamental rights\b|\bhuman dignity\b", True),
    ("value", "deregulation", r"\bderegulat(?:e|ion|ory)\b|\bregulatory easing\b|\bremoving barriers\b", True),
    ("value", "voluntary/self-regulation", r"\bvoluntary commitment(?:s)?\b|\bself-regulat(?:e|ion|ory)\b|\bindustry commitment(?:s)?\b", True),
    ("value", "AI dominance/leadership", r"\btechnological supremacy\b|\bAI dominance\b|\bAI leadership\b|\bglobal leadership\b", True),
    ("value", "competitiveness", r"\b(?:economic|industrial|national|global|technological|ai) competitiveness\b|\bcompetitive advantage\b", True),
    ("value", "strategic autonomy", r"\bstrategic autonomy\b|\btechnological autonomy\b|\bdigital autonomy\b", True),
    ("value", "technology self-reliance", r"\btechnological self[- ]reliance\b|\btechnology self[- ]sufficiency\b|\bindigenous innovation\b", True),
    ("value", "responsible innovation", r"\bresponsible innovation\b", True),
    ("value", "trustworthy AI", r"\btrustworthy (?:ai|artificial intelligence)\b|\btrusted ai\b", True),
    ("value", "human-centered AI", r"\bhuman[- ]cent(?:ered|red) (?:ai|artificial intelligence|development|governance)\b", True),
    ("value", "AI for good", r"\bai for good\b|\btechnology for good\b|\bbenefit human(?:ity|kind)\b", True),
    ("value", "social stability/collective benefit", r"\bsocial stability\b|\bsocial harmony\b|\bcollective (?:benefit|welfare)\b|\bcommon prosperity\b|\bcommon good\b", True),
    ("value", "state-led development", r"\b(?:state|government)[- ](?:led|directed|driven)\b|\bnational champion(?:s)?\b|\bindustrial policy\b", True),
    ("value", "core socialist values", r"\bcore socialist values\b", True),
    ("value", "American values", r"\bamerican values\b", True),
    ("value", "Chinese values/characteristics", r"\bchinese values\b|\bchinese characteristics\b|\bsocialism with chinese characteristics\b", True),
    ("value", "civil rights/liberties", r"\bcivil rights?\b|\bcivil libert(?:y|ies)\b", True),
    # Keep responsibility specific: the bare word is too polysemous to interpret.
    ("value", "developer responsibility", r"\b(?:developer|provider|company|firm|industry) responsibilit(?:y|ies)\b|\bduty of care\b|\bresponsible (?:AI|development|deployment|scaling)\b", True),
    # -- F. geopolitical / competition framing -------------------------------------
    ("geopolitical", "national security", r"\bnational security\b", True),
    ("geopolitical", "arms race",    r"\barms race\b", True),
    ("geopolitical", "competition/race", r"\b(?:geopolitical|global|technological) (?:competition|race)\b|\brace to the bottom\b", True),
    ("geopolitical", "surveillance", r"\bsurveillance\b", True),
    ("geopolitical", "censorship",   r"\bcensor(?:ship|ed|ing)?\b", True),
    ("geopolitical", "small yard high fence", r"\bsmall yard[,\s]+high fence\b|\byard and fence\b", True),
    ("geopolitical", "decoupling",   r"\bdecoupl(?:e|ing)\b", True),
    ("geopolitical", "chokepoint",   r"\bchoke[-\s]?point(?:s)?\b", True),
    ("geopolitical", "hegemony",     r"\bhegemon(?:y|ic)\b", True),
    ("geopolitical", "containment",  r"\bcontainment\b", True),
    ("geopolitical", "level playing field", r"\blevel playing field\b", True),
    ("geopolitical", "international public good", r"\binternational public good\b", True),
    ("geopolitical", "unilateralism", r"\bunilateral(?:ism)?\b", True),
    ("geopolitical", "allies/like-minded partners", r"\b(?:allies|allied democracies|like-minded (?:partners|countries|states)|trusted partners)\b", True),
    ("geopolitical", "foreign adversaries", r"\bforeign adversar(?:y|ies)\b|\badversarial (?:state|nation|actor)s?\b|\bhostile (?:state|actor)s?\b", True),
    ("geopolitical", "China denial/restriction", r"\b(?:prevent|deny|restrict|limit|block|keep)\s+(?:china|chinese (?:firms|companies|labs|developers))\b", True),
    ("geopolitical", "strategic competition", r"\bstrategic competition\b|\bstrategic rival(?:ry)?\b", True),
    ("cn_rhetoric", "developing countries/Global South", r"\bdeveloping (?:countries|economies|nations)\b|\bglobal south\b", True),
    ("cn_rhetoric", "digital divide", r"\bdigital divide\b|\bai divide\b|\bintelligence divide\b", True),
    ("geopolitical", "deepfake",     r"\bdeep[- ]?fake(?:s)?\b|\bsynthetic media\b", True),
    # -- F2. loaded diplomatic rhetoric (CN + Western), for bilingual comparison -----
    ("cn_rhetoric", "Cold War mentality", r"\bcold war mentality\b|\bcold war thinking\b", True),
    ("cn_rhetoric", "zero-sum", r"\bzero[- ]sum\b", True),
    ("cn_rhetoric", "technology blockade", r"\btechnolog(?:y|ical) blockade\b|\btechnology embargo\b", True),
    ("cn_rhetoric", "over-securitization", r"\bover[- ]securiti[sz]ation\b|\bsecuriti[sz]ing everything\b", True),
    ("cn_rhetoric", "double standards", r"\bdouble standards?\b", True),
    ("cn_rhetoric", "exclusive blocs", r"\bexclusive blocs?\b|\bbloc politics\b|\bcamp confrontation\b|\bsmall circles\b|\bclique politics\b", True),
    ("cn_rhetoric", "community with a shared future", r"\bcommunity with a shared future for (?:mankind|humanity)\b|\bcommunity of common destiny\b", True),
    ("cn_rhetoric", "true multilateralism", r"\b(?:true|genuine) multilateralism\b", True),
    ("cn_rhetoric", "inclusive and mutually beneficial", r"\binclusive development\b|\bmutual benefit\b|\bwin[- ]win cooperation\b", True),
    ("us_rhetoric", "rules-based order", r"\brules[- ]based (?:international|global) order\b", True),
    ("us_rhetoric", "democratic values/free world", r"\bdemocratic values\b|\bfree world\b|\balliance of democracies\b|\bdemocratic nations\b", True),
    # -- G. risk register ----------------------------------------------------------
    ("risk", "catastrophic",         r"\bcatastroph(?:e|es|ic|ically)\b", True),
    ("risk", "existential",          r"\bexistential\b", True),
    # bare "high-risk" dropped: it denotes the EU AI Act risk-tier label, not a
    # catastrophic register (validation false positives V0001/V0002).  Expanded to the
    # frontier / loss-of-control vocabulary the frame definition already claims.
    ("risk", "high/systemic risk",
     r"\bsystemic risk\b|\bloss[-\s]?of[-\s]?control\b"
     r"|\buncontroll(?:able|ed)\b|\brogue AI\b|\bhuman extinction\b|\bextinction[-\s]?level\b"
     r"|\bthreat to (?:humanity|human survival)\b|\bend of humanity\b|\bcivili[sz]ational\b", True),
    # -- H. openness / access framing ----------------------------------------------
    ("access", "open weights",       r"\bopen[- ]weight(?:s|ed)?\b|\bpublicly (?:released|available) weights\b|\brelease (?:of )?(?:the )?model weights\b", True),
    ("access", "open source",        r"\bopen[- ]source\b|\bopen source ai\b", True),
    # -- I. regulatory approach (added from Prenga 2024 keyness + main.tex appendix) --
    ("approach", "ex-ante / pre-deployment", r"\bex[- ]ante\b|\bpre[- ]deployment\b|\bpre[- ]market\b|\bpreclearance\b|\bpre[- ]approval\b|\bprior (?:approval|authorization)\b|\badvance approval\b", True),
    ("approach", "ex-post / after-the-fact", r"\bex[- ]post\b|\bpost[- ]deployment (?:oversight|regulation|liability)\b|\bafter[- ]the[- ]fact\b|\bliability after harm\b|\bretrospective regulation\b", True),
    ("approach", "risk-based regulation", r"\brisk[- ]based (?:regulation|governance|framework|approach)\b|\btiered risk (?:framework|regulation|classification)\b", True),
    ("approach", "regulatory sandbox", r"\bregulatory sandbox(?:es)?\b|\bsandbox regulation\b", True),
    ("approach", "conformity assessment", r"\bconformity assessment\b", True),
    ("approach", "general purpose AI (GPAI)", r"(?i:\bgeneral[-\s]purpose ai\b)|\bGPAI\b", False),
    ("approach", "market surveillance", r"\bmarket surveillance\b", True),
    ("approach", "horizontal regulation", r"\bhorizontal (?:regulation|framework|regime|rules|legislation)\b", True),
    ("approach", "vertical/sector-specific", r"\bvertical (?:regulation|framework|regime|rules|legislation)\b|\bsector[- ]specific (?:regulation|rules|oversight|approach)\b", True),
    # -- K. named instruments (added) -------------------------------------------------
    ("institution", "Digital Omnibus", r"\bdigital omnibus\b", True),
    ("institution", "AI Action Plan", r"\bAI Action Plan\b|\bNational (?:AI )?Policy Framework\b", False),
    ("institution", "AIDP/New Gen AI Dev Plan", r"\bNew Generation (?:Artificial Intelligence )?Development Plan\b|\bAIDP\b", False),
    # -- M. developer / market structure -------------------------------------------
    ("industry", "strengthen export controls", r"\b(?:strengthen|tighten|expand|update|enforce)\s+(?:export controls?|export restrictions?)\b", True),
    ("industry", "export-control evasion", r"\bexport[- ]control (?:evasion|circumvention)\b|\b(?:evade|circumvent|bypass) export (?:controls?|restrictions?)\b|\bchip smuggling\b|\bsemiconductor diversion\b", True),
    ("industry", "compute/chip governance", r"\bcompute (?:governance|thresholds?|access)\b|\badvanced (?:ai )?(?:chips|semiconductors)\b", True),
    ("industry", "frontier capability thresholds", r"\bfrontier (?:ai|model)s?\b|\b(?:capability|compute) threshold(?:s)?\b|\bdangerous capabilit(?:y|ies)\b", True),
    ("industry", "lock-in/concentration", r"\b(?:vendor|platform|ecosystem) lock[-\s]?in\b|\bmarket concentration\b|\b(?:monopoly|oligopoly)\b", True),
    ("industry", "closed/proprietary models", r"\bclosed[-\s]?source\b|\bproprietary (?:model|system|ai)\b", True),
]


def compile_terms():
    out = []
    for cat, label, pat, ic in TERMS:
        flags = re.IGNORECASE if ic else 0
        out.append((cat, label, re.compile(pat, flags)))
    return out


def build_matches(resp: pd.DataFrame) -> pd.DataFrame:
    """One row per (response, term): match count. Plus a `words` column on resp."""
    terms = compile_terms()
    resp = resp.copy()
    resp["words"] = resp["response"].str.split().str.len().clip(lower=1)
    counts = {}
    for cat, label, rx in terms:
        counts[label] = resp["response"].apply(lambda t: len(rx.findall(t)))
    m = pd.DataFrame(counts, index=resp.index)
    return resp, m, terms


def share_rate_table(resp: pd.DataFrame, m: pd.DataFrame, terms) -> pd.DataFrame:
    """Long: category, term, family, model, n, n_mention, share, matches, words, rate."""
    cat_of = {label: cat for cat, label, _ in terms}
    rows = []
    for fam in ["A", "B", "C", "ALL"]:
        sub_idx = resp.index if fam == "ALL" else resp.index[resp["family"] == fam]
        for model in MODEL_ORDER:
            mi = sub_idx[resp.loc[sub_idx, "model"] == model]
            if len(mi) == 0:
                continue
            words = resp.loc[mi, "words"].sum()
            for label in m.columns:
                cnt = m.loc[mi, label]
                rows.append({
                    "category": cat_of[label], "term": label, "family": fam,
                    "model": model, "origin": MODEL_ORIGIN[model],
                    "n": len(mi), "n_mention": int((cnt > 0).sum()),
                    "share": round(100 * (cnt > 0).mean(), 2),
                    "matches": int(cnt.sum()), "words": int(words),
                    "rate_per_1k": round(1000 * cnt.sum() / words, 3),
                })
    return pd.DataFrame(rows)


def origin_tests(resp: pd.DataFrame, m: pd.DataFrame, terms) -> pd.DataFrame:
    """Per (term, family): logistic mention~C(origin), issue-clustered; joint Wald p."""
    cat_of = {label: cat for cat, label, _ in terms}
    rows = []
    for fam in ["A", "B", "C"]:
        idx = resp.index[resp["family"] == fam]
        d0 = resp.loc[idx, ["origin", "issue_id"]].copy()
        for label in m.columns:
            d = d0.copy()
            d["y"] = (m.loc[idx, label] > 0).astype(int)
            p = np.nan
            # need >=2 origins with variation and both outcomes present
            if 0 < d["y"].mean() < 1 and d["origin"].nunique() >= 2:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = smf.logit("y ~ C(origin)", data=d).fit(
                            disp=0, cov_type="cluster",
                            cov_kwds={"groups": d["issue_id"]}, maxiter=100)
                    terms_idx = [i for i, n in enumerate(fit.params.index)
                                 if n.startswith("C(origin)")]
                    if terms_idx:
                        R = np.zeros((len(terms_idx), len(fit.params)))
                        for r, ti in enumerate(terms_idx):
                            R[r, ti] = 1
                        p = float(fit.wald_test(R, scalar=True).pvalue)
                except Exception:
                    p = np.nan
            rows.append({"category": cat_of[label], "term": label, "family": fam,
                         "n_issue_clusters": d["issue_id"].nunique(),
                         "overall_share": round(100 * d["y"].mean(), 2),
                         "origin_wald_p": p})
    out = pd.DataFrame(rows)
    # BH-FDR within family, over the terms that produced a p-value
    out["p_bh"] = np.nan
    for fam in ["A", "B", "C"]:
        mask = (out["family"] == fam) & out["origin_wald_p"].notna()
        pv = out.loc[mask, "origin_wald_p"].values
        if len(pv):
            order = np.argsort(pv)
            ranked = pv[order]
            n = len(pv)
            bh = ranked * n / (np.arange(1, n + 1))
            bh = np.minimum.accumulate(bh[::-1])[::-1].clip(max=1)
            adj = np.empty(n)
            adj[order] = bh
            out.loc[mask, "p_bh"] = np.round(adj, 4)
    return out


def fig_share_heatmap(sr: pd.DataFrame, fam: str) -> None:
    """Share (model x term) heatmap for one family; terms ordered by category."""
    d = sr[(sr["family"] == fam)]
    order = [t for _, t, _ in compile_terms() if t in set(d["term"])]
    piv = d.pivot_table(index="term", columns="model", values="share").reindex(
        index=order, columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(order) + 1.2))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(order)), order, fontsize=7)
    for i in range(len(order)):
        for j in range(len(MODEL_ORDER)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 55 else INK)
    ax.set_title(f"Term SHARE (% of responses, EN) — {FAMILY_LABEL[fam]}",
                 fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% responses")
    save(fig, FIG / f"nb5d_share_heatmap_{fam}")


def top_phrases_table(resp: pd.DataFrame, group_col: str, groups: list,
                       ngram_range=(2, 3), topn=20) -> pd.DataFrame:
    """Most common 2-3 word phrases per group (model or origin), by family.

    Data-driven complement to the curated TERMS list -- not filtered against it,
    so expect topical n-grams (e.g. "facial recognition") alongside framing ones;
    read alongside the per-issue TF-IDF scratch file for the same caveat (proper
    nouns / hallucinated abbreviations can surface, especially at low min_df).
    """
    rows = []
    for fam in ["A", "B", "C"]:
        sub = resp[resp.family == fam]
        for g in groups:
            texts = sub.loc[sub[group_col] == g, "response"]
            if len(texts) < 10:
                continue
            vec = CountVectorizer(
                lowercase=True, stop_words="english", ngram_range=ngram_range,
                token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}",
                min_df=max(5, int(0.01 * len(texts))), max_df=0.9)
            try:
                X = vec.fit_transform(texts)
            except ValueError:
                continue
            counts = np.asarray(X.sum(axis=0)).ravel()
            vocab = np.array(vec.get_feature_names_out())
            doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
            top_idx = np.argsort(counts)[::-1][:topn]
            for rank, ti in enumerate(top_idx, start=1):
                rows.append({
                    "family": fam, group_col: g, "rank": rank, "phrase": vocab[ti],
                    "count": int(counts[ti]), "n_docs_with_phrase": int(doc_freq[ti]),
                    "n_docs": len(texts),
                    "share_docs": round(100 * doc_freq[ti] / len(texts), 2),
                })
    return pd.DataFrame(rows)


def unique_phrases_by_model(resp: pd.DataFrame, ngram_range=(2, 3), topn=15,
                             min_share=5.0, max_df=0.5) -> pd.DataFrame:
    """Phrases distinctive to ONE model, not just common ones every model uses.

    `top_phrases_table` ranks by raw count within a group, which surfaces shared
    topical vocabulary (training data, artificial intelligence) that every model
    uses -- that's expected, not a finding. Here the vocabulary is fit ONCE per
    family across all six models pooled (so "share" is comparable across
    models), max_df=0.5 drops anything more than half the pooled documents use
    (screens out the near-universal topical phrases), and phrases are ranked by
    OVER-INDEX: this model's % of docs containing the phrase minus the MEAN of
    the other five models' % (mirrors the SIGNATURE calculation already used for
    the curated TERMS list, so the two are read the same way). min_share floors
    out noise from rare phrases that "win" only because every model's count is
    near zero. Done by MODEL rather than origin: origin groups are 3/2/1 models
    and the keyword-share analysis above already shows origin is a weak,
    inconsistent sorter of vocabulary -- the model IS the unit that carries a
    real signature.
    """
    rows = []
    for fam in ["A", "B", "C"]:
        sub = resp[resp.family == fam]
        vec = CountVectorizer(
            lowercase=True, stop_words="english", ngram_range=ngram_range,
            token_pattern=r"[a-zA-Z][a-zA-Z\-]{2,}",
            min_df=max(10, int(0.005 * len(sub))), max_df=max_df, binary=True)
        try:
            X = vec.fit_transform(sub["response"])
        except ValueError:
            continue
        vocab = vec.get_feature_names_out()
        models = sub["model"].to_numpy()
        share = {}
        for mdl in MODEL_ORDER:
            mask = models == mdl
            if mask.sum() == 0:
                continue
            share[mdl] = 100 * np.asarray(X[mask].sum(axis=0)).ravel() / mask.sum()
        for mdl in MODEL_ORDER:
            if mdl not in share:
                continue
            others = np.vstack([share[m2] for m2 in MODEL_ORDER if m2 != mdl]).mean(axis=0)
            over_index = share[mdl] - others
            elig = share[mdl] >= min_share
            order = np.argsort(np.where(elig, over_index, -np.inf))[::-1][:topn]
            for rank, ti in enumerate(order, start=1):
                if not elig[ti]:
                    continue
                rows.append({
                    "family": fam, "model": mdl, "origin": MODEL_ORIGIN[mdl],
                    "rank": rank, "phrase": vocab[ti],
                    "share_this_model": round(float(share[mdl][ti]), 2),
                    "share_other_mean": round(float(others[ti]), 2),
                    "over_index": round(float(over_index[ti]), 2),
                })
    return pd.DataFrame(rows)


def run_keyword_framing_en() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    resp = load_responses()
    resp = resp[resp["lang"] == "en"].copy()
    resp["origin"] = resp["model"].map(MODEL_ORIGIN)

    master = load_master()
    fam_of = master.drop_duplicates("issue_id").set_index("issue_id")["family"].to_dict()
    resp["family"] = resp["issue_id"].map(fam_of)

    refused = set(map(tuple, master.loc[master["is_refusal"] == 1,
                                        ["rep", "model", "prompt_id"]].values))
    keys = list(map(tuple, resp[["rep", "model", "prompt_id"]].values))
    resp = resp[[k not in refused for k in keys]].reset_index(drop=True)
    print(f"{len(resp):,} non-refusal EN responses "
          f"(A={sum(resp.family=='A'):,} B={sum(resp.family=='B'):,} "
          f"C={sum(resp.family=='C'):,})\n")

    resp, m, terms = build_matches(resp)

    sr = share_rate_table(resp, m, terms)
    sr.to_csv(TAB / "nb5d_keyword_share_rate.csv", index=False)

    tests = origin_tests(resp, m, terms)
    tests.to_csv(TAB / "nb5d_origin_tests.csv", index=False)

    for fam in ["A", "B", "C"]:
        fig_share_heatmap(sr, fam)

    phrases_model = top_phrases_table(resp, "model", MODEL_ORDER)
    phrases_model.to_csv(TAB / "nb5d_top_phrases_model.csv", index=False)
    phrases_origin = top_phrases_table(resp, "origin", ORIGIN_ORDER)
    phrases_origin.to_csv(TAB / "nb5d_top_phrases_origin.csv", index=False)

    unique_model = unique_phrases_by_model(resp)
    unique_model.to_csv(TAB / "nb5d_unique_phrases_model.csv", index=False)

    # console: per-model fingerprint on the contested B-series. Lead with each model's
    # SIGNATURE -- the terms it over-indexes on relative to the other five models -- so
    # a model that is quiet in absolute terms (Llama) still gets a real profile rather
    # than "nothing". Effect size (the share gap) is the story here, not the p-value.
    print("=== SHARE (%) on B-series, by model ===")
    piv = sr[sr.family == "B"].pivot_table(index="term", columns="model",
                                           values="share").reindex(columns=MODEL_ORDER)
    order = [t for _, t, _ in terms if t in piv.index]
    print(piv.reindex(order).to_string(float_format=lambda v: f"{v:5.1f}"))

    print("\n=== per-model SIGNATURE on B (top over-index vs. the other 5 models) ===")
    for model in MODEL_ORDER:
        others = [c for c in MODEL_ORDER if c != model]
        rel = (piv[model] - piv[others].mean(axis=1)).dropna().sort_values(ascending=False)
        tags = [f"{t} {piv.loc[t, model]:.0f}% (Δ{rel[t]:+.0f})" for t in rel.head(5).index]
        print(f"  {model:9s} {MODEL_ORIGIN[model]}: " + "; ".join(tags))

    print("\n=== top phrases (2-3 grams) on B-series, by ORIGIN ===")
    pb = phrases_origin[phrases_origin.family == "B"]
    for g in ORIGIN_ORDER:
        top = pb[pb.origin == g].sort_values("rank").head(10)
        print(f"  {g}: " + "; ".join(
            f"{r.phrase} ({r.count}, {r.share_docs:.0f}% docs)" for r in top.itertuples()))

    print("\n=== top phrases (2-3 grams) on B-series, by MODEL ===")
    pm = phrases_model[phrases_model.family == "B"]
    for g in MODEL_ORDER:
        top = pm[pm.model == g].sort_values("rank").head(10)
        print(f"  {g:9s} {MODEL_ORIGIN[g]}: " + "; ".join(
            f"{r.phrase} ({r.count}, {r.share_docs:.0f}% docs)" for r in top.itertuples()))

    print("\n=== UNIQUE phrases on B-series, by MODEL (over-index vs mean of other 5) ===")
    ub = unique_model[unique_model.family == "B"]
    for g in MODEL_ORDER:
        top = ub[ub.model == g].sort_values("rank").head(10)
        print(f"  {g:9s} {MODEL_ORIGIN[g]}: " + "; ".join(
            f"{r.phrase} ({r.share_this_model:.0f}% vs {r.share_other_mean:.0f}%, "
            f"Δ+{r.over_index:.0f})" for r in top.itertuples()))

    print(f"\ntables -> {TAB}/nb5d_keyword_share_rate.csv")
    print(f"         {TAB}/nb5d_origin_tests.csv")
    print(f"         {TAB}/nb5d_top_phrases_model.csv")
    print(f"         {TAB}/nb5d_top_phrases_origin.csv")
    print(f"         {TAB}/nb5d_unique_phrases_model.csv")
    print(f"figs   -> {FIG}/nb5d_share_heatmap_[A|B|C].pdf|png")




# --- nb5d_zh_keyword_framing.py ---

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "evaluation/analysis")
sys.path.insert(0, "evaluation/scripts/06_analysis")
from analysis_helpers import load_master, load_responses  # noqa: E402
from viz_style import INK, MODEL_ORDER, MODEL_ORIGIN, SEQUENTIAL, apply_style, save  # noqa: E402

FIG = Path("evaluation/analysis/outputs/figures")
TAB = Path("evaluation/analysis/outputs/tables")
FAMILY_LABEL = {"A": "A baselines", "B": "B contested", "C": "C named"}


def compile_terms(terms):
    # ZH: substring match on raw text, no \b (CJK has no word boundaries). Patterns that
    # embed latin abbreviations (GDPR, CAC) still work as plain substrings.
    return [(cat, label, re.compile(pat)) for cat, label, pat, *_ in terms]


def build_matches(resp: pd.DataFrame, terms):
    """One column per term = match count per response. resp gains jieba word count."""
    import jieba
    resp = resp.copy()
    resp["words"] = resp["response"].map(
        lambda t: max(sum(1 for w in jieba.cut(str(t)) if w.strip()), 1))
    counts = {label: resp["response"].apply(lambda t: len(rx.findall(str(t))))
              for _, label, rx in terms}
    return resp, pd.DataFrame(counts, index=resp.index)


def canon(label: str) -> str:
    """Concept key for EN<->ZH join: drop a trailing ' (漢字)' annotation, lowercase."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip().lower()


def fig_share_heatmap(sr: pd.DataFrame, order, fam: str, min_share: float = 5.0) -> None:
    d = sr[sr["family"] == fam]
    piv = d.pivot_table(index="term", columns="model", values="share").reindex(
        index=[t for t in order if t in set(d["term"])], columns=MODEL_ORDER)
    piv = piv[piv.max(axis=1) >= min_share]  # drop rare terms so the figure stays legible
    # display label: drop the trailing " (漢字)" annotation so no CJK glyphs are needed
    ylabels = [re.sub(r"\s*\([^)]*\)\s*$", "", t) for t in piv.index]
    fig, ax = plt.subplots(figsize=(7.6, 0.30 * len(piv) + 1.2))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(piv)), ylabels, fontsize=6.5)
    for i in range(len(piv)):
        for j in range(len(MODEL_ORDER)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > 55 else INK)
    ax.set_title(f"ZH term SHARE (% of responses) — {FAMILY_LABEL[fam]}",
                 fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% responses")
    save(fig, FIG / f"nb5d_zh_share_heatmap_{fam}")


def run_keyword_framing_zh() -> None:
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    resp = load_responses()
    resp = resp[resp["lang"] == "zh"].copy()
    resp["origin"] = resp["model"].map(MODEL_ORIGIN)
    master = load_master()
    fam_of = master.drop_duplicates("issue_id").set_index("issue_id")["family"].to_dict()
    resp["family"] = resp["issue_id"].map(fam_of)
    refused = set(map(tuple, master.loc[master["is_refusal"] == 1,
                                        ["rep", "model", "prompt_id"]].values))
    keys = list(map(tuple, resp[["rep", "model", "prompt_id"]].values))
    resp = resp[[k not in refused for k in keys]].reset_index(drop=True)
    print(f"{len(resp):,} non-refusal ZH responses "
          f"(A={sum(resp.family=='A'):,} B={sum(resp.family=='B'):,} "
          f"C={sum(resp.family=='C'):,})\n")

    terms = compile_terms(ZH_TERMS)
    resp, m = build_matches(resp, terms)

    sr = share_rate_table(resp, m, terms)
    sr.to_csv(TAB / "nb5d_zh_keyword_share_rate.csv", index=False)
    tests = origin_tests(resp, m, terms)
    tests.to_csv(TAB / "nb5d_zh_origin_tests.csv", index=False)

    order = [label for _, label, _ in terms]
    for fam in ["A", "B", "C"]:
        fig_share_heatmap(sr, order, fam)

    # ---- EN <-> ZH comparison on shared concepts (B-series share) --------------------
    en_sr = pd.read_csv(TAB / "nb5d_keyword_share_rate.csv")
    en_b = en_sr[en_sr.family == "B"].groupby("term").apply(
        lambda g: 100 * g.n_mention.sum() / g.n.sum(), include_groups=False)
    zh_b = sr[sr.family == "B"].groupby("term").apply(
        lambda g: 100 * g.n_mention.sum() / g.n.sum(), include_groups=False)
    en_key = {canon(t): (t, v) for t, v in en_b.items()}
    zh_key = {canon(t): (t, v) for t, v in zh_b.items()}
    shared = sorted(set(en_key) & set(zh_key))
    rows = [{"concept": k, "en_term": en_key[k][0], "zh_term": zh_key[k][0],
             "en_share_B": round(en_key[k][1], 2), "zh_share_B": round(zh_key[k][1], 2),
             "zh_minus_en": round(zh_key[k][1] - en_key[k][1], 2)} for k in shared]
    comp = pd.DataFrame(rows).sort_values("zh_minus_en")
    comp.to_csv(TAB / "nb5d_en_zh_comparison.csv", index=False)

    print(f"=== EN<->ZH shared concepts: {len(shared)} matched "
          f"(EN-only {len(set(en_key)-set(zh_key))}, ZH-only {len(set(zh_key)-set(en_key))}) ===")
    print("  biggest ZH>EN (more salient in Chinese):")
    for r in comp.tail(10).iloc[::-1].itertuples():
        print(f"    {r.concept:34s} ZH {r.zh_share_B:5.1f}  EN {r.en_share_B:5.1f}  Δ{r.zh_minus_en:+.1f}")
    print("  biggest EN>ZH (more salient in English):")
    for r in comp.head(8).itertuples():
        print(f"    {r.concept:34s} ZH {r.zh_share_B:5.1f}  EN {r.en_share_B:5.1f}  Δ{r.zh_minus_en:+.1f}")

    # unmatched labels, so we can align them later
    en_only = sorted(set(en_key) - set(zh_key))
    zh_only = sorted(set(zh_key) - set(en_key))
    pd.DataFrame({"en_only": pd.Series(en_only)}).to_csv(TAB / "nb5d_en_only_concepts.csv", index=False)
    pd.DataFrame({"zh_only": pd.Series(zh_only)}).to_csv(TAB / "nb5d_zh_only_concepts.csv", index=False)

    print("\n=== per-model SIGNATURE on B (ZH; top over-index vs other 5) ===")
    piv = sr[sr.family == "B"].pivot_table(index="term", columns="model",
                                           values="share").reindex(columns=MODEL_ORDER)
    for model in MODEL_ORDER:
        others = [c for c in MODEL_ORDER if c != model]
        rel = (piv[model] - piv[others].mean(axis=1)).dropna().sort_values(ascending=False)
        tags = [f"{t} {piv.loc[t, model]:.0f}%(Δ{rel[t]:+.0f})" for t in rel.head(6).index]
        print(f"  {model:9s} {MODEL_ORIGIN[model]}: " + "; ".join(tags))

    print(f"\ntables -> {TAB}/nb5d_zh_keyword_share_rate.csv, nb5d_zh_origin_tests.csv,")
    print(f"         {TAB}/nb5d_en_zh_comparison.csv (+ *_only_concepts.csv)")
    print(f"figs   -> {FIG}/nb5d_zh_share_heatmap_[A|B|C].pdf|png")




# --- nb5i_stance_conditioned_framing.py ---


import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "evaluation/analysis")
sys.path.insert(0, "evaluation/scripts/06_analysis")
from analysis_helpers import load_master, load_responses  # noqa: E402


TAB = Path("evaluation/analysis/outputs/tables")
REVIEW = Path("evaluation/analysis/outputs/manual_coding")
SEED = 20260725
OPENING_FRACTION = 0.20
# The validation sample is a pure random draw over the whole corpus, with no issue held
# out: it certifies the instrument against exactly the data the instrument is applied to,
# so excluding any issue (even one already hand-read for RQ4) would bias the estimate.
VALIDATION_EXCLUDE_ISSUES: set[str] = set()
MODEL_ORIGIN = {
    "claude": "US", "gpt4o": "US", "llama": "US",
    "deepseek": "CN", "qwen": "CN", "mistral": "EU",
}


def load_en_terms_literal():
    return TERMS


def compile_raw_terms(lang: str) -> dict[str, re.Pattern]:
    """Compile only the raw terms required by the core frame aggregation."""
    needed = {x for spec in FRAME_COMPONENTS.values() for x in spec[lang]}
    out = {}
    if lang == "en":
        rows = load_en_terms_literal()
        for _, label, pattern, ignorecase in rows:
            if label in needed:
                out[label] = re.compile(pattern, re.IGNORECASE if ignorecase else 0)
    else:
        rows = ZH_TERMS
        for _, label, pattern, *_ in rows:
            if label in needed:
                # Do not let 国家安全 automatically count as the general safety frame.
                if label == "safety":
                    pattern = r"(?<!国家)安全"
                out[label] = re.compile(pattern)
    missing = sorted(needed - set(out))
    if missing:
        raise ValueError(f"{lang}: frame components missing from raw lexicon: {missing}")
    return out


def compile_direct(patterns: dict, lang: str) -> dict[str, re.Pattern]:
    flags = re.IGNORECASE if lang == "en" else 0
    return {label: re.compile(spec[lang], flags) for label, spec in patterns.items()}


def opening_segment(text: str) -> str:
    """First fifth of response characters, a transparent measure of foregrounding."""
    text = str(text)
    return text[: max(1, int(np.ceil(len(text) * OPENING_FRACTION)))]


def paragraph_lead_segments(text: str) -> str:
    """Join the first sentence of every substantive paragraph/block.

    This captures topic-sentence salience after a generic introduction.  Markdown
    headings and bullet blocks count as substantive blocks because models often use them
    to announce the lens of the section that follows.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", str(text)) if b.strip()]
    leads = []
    for block in blocks:
        match = re.search(r"[.!?。！？](?:[\"'”’）)])?", block)
        leads.append(block[:match.end()] if match else block[:300])
    return "\n".join(leads)


def first_excerpt(text: str, regexes: list[re.Pattern], radius: int = 280) -> str:
    """Return context around the first matching component, or the opening if no hit."""
    text = str(text)
    hits = [m for rx in regexes if (m := rx.search(text)) is not None]
    if not hits:
        return text[: 2 * radius].replace("\n", " ")
    m = min(hits, key=lambda x: x.start())
    lo, hi = max(0, m.start() - radius), min(len(text), m.end() + radius)
    prefix = "…" if lo else ""
    suffix = "…" if hi < len(text) else ""
    return (prefix + text[lo:hi] + suffix).replace("\n", " ")


def match_constructs(texts: pd.Series, lang: str):
    """Return wide presence/opening/paragraph-lead matrices and match metadata."""
    text_list = texts.fillna("").astype(str).tolist()
    opening_list = [opening_segment(x) for x in text_list]
    paragraph_leads = [paragraph_lead_segments(x) for x in text_list]
    raw = compile_raw_terms(lang)

    counts, foreground, lead_sentence, components, regexes = {}, {}, {}, {}, {}

    # Frames aggregate component-term counts.
    for construct, spec in FRAME_COMPONENTS.items():
        labels = spec[lang]
        rxs = [raw[label] for label in labels]
        # One combined scan per construct is substantially faster than rescanning every
        # long response once per component.  Individual regexes are retained for the
        # validation workbook so reviewers can see which component fired.
        combined = re.compile(
            "|".join(f"(?:{rx.pattern})" for rx in rxs),
            re.IGNORECASE if lang == "en" else 0,
        )
        full = np.fromiter(
            (combined.search(x) is not None for x in text_list),
            dtype=np.int8, count=len(text_list),
        )
        fore = np.fromiter(
            (combined.search(x) is not None for x in opening_list),
            dtype=np.int8, count=len(opening_list),
        )
        lead = np.fromiter(
            (combined.search(x) is not None for x in paragraph_leads),
            dtype=np.int8, count=len(paragraph_leads),
        )
        key = ("frame", construct)
        counts[key], foreground[key], lead_sentence[key] = full, fore, lead
        components[key], regexes[key] = labels, rxs

    # Actor, mechanism, and risk dimensions use direct construct patterns.
    for dimension, source in [
        ("actor", ACTOR_PATTERNS),
        ("mechanism", MECHANISM_PATTERNS),
        ("risk_type", RISK_PATTERNS),
    ]:
        for construct, rx in compile_direct(source, lang).items():
            key = (dimension, construct)
            counts[key] = np.fromiter(
                (rx.search(x) is not None for x in text_list),
                dtype=np.int8, count=len(text_list),
            )
            foreground[key] = np.fromiter(
                (rx.search(x) is not None for x in opening_list),
                dtype=np.int8, count=len(opening_list),
            )
            lead_sentence[key] = np.fromiter(
                (rx.search(x) is not None for x in paragraph_leads),
                dtype=np.int8, count=len(paragraph_leads),
            )
            components[key], regexes[key] = [construct], [rx]

    columns = pd.MultiIndex.from_tuples(counts, names=["dimension", "construct"])
    count_df = pd.DataFrame(np.column_stack(list(counts.values())), columns=columns)
    fore_df = pd.DataFrame(np.column_stack(list(foreground.values())), columns=columns)
    lead_df = pd.DataFrame(np.column_stack(list(lead_sentence.values())), columns=columns)
    return count_df, fore_df, lead_df, components, regexes


def load_analysis_rows() -> pd.DataFrame:
    # keep only the merge keys + text from responses; template_id/family/stance come from
    # master, so subsetting here avoids a template_id_x/_y column collision in the merge.
    response_cols = ["rep", "model", "prompt_id", "issue_id", "lang", "response"]
    responses = load_responses()[response_cols].copy()
    master = load_master()
    meta_cols = [
        "rep", "model", "prompt_id", "issue_id", "family", "template_id", "lang",
        "stance_num", "is_refusal",
    ]
    meta = master[meta_cols].drop_duplicates(["rep", "model", "prompt_id"])
    rows = responses.merge(
        meta, on=["rep", "model", "prompt_id", "issue_id", "lang"], how="inner",
        validate="one_to_one",
    )
    required = set(response_cols + meta_cols)
    missing = sorted(required - set(rows.columns))
    if missing:
        raise AssertionError(f"Merged framing rows missing required columns: {missing}")
    rows = rows[(rows["is_refusal"] == 0) & rows["stance_num"].notna()].copy()
    rows["origin"] = rows["model"].map(MODEL_ORIGIN)
    rows["response_chars"] = rows["response"].astype(str).str.len()
    return rows.reset_index(drop=True)


def prevalence_table(meta: pd.DataFrame, count: pd.DataFrame,
                     fore: pd.DataFrame, lead: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["family", "lang", "model", "origin"]
    for keys, idx in meta.groupby(group_cols, observed=True).indices.items():
        for dimension, construct in count.columns:
            vals = count.loc[idx, (dimension, construct)].to_numpy()
            front = fore.loc[idx, (dimension, construct)].to_numpy()
            topic_lead = lead.loc[idx, (dimension, construct)].to_numpy()
            rows.append({
                **dict(zip(group_cols, keys)), "dimension": dimension,
                "construct": construct, "n": len(idx),
                "mention_share": float(np.mean(vals > 0)),
                "foreground_share": float(np.mean(front > 0)),
                "foreground_given_mention": (
                    float(np.mean(front[vals > 0] > 0)) if np.any(vals > 0) else np.nan
                ),
                "paragraph_lead_share": float(np.mean(topic_lead > 0)),
                "paragraph_lead_given_mention": (
                    float(np.mean(topic_lead[vals > 0] > 0))
                    if np.any(vals > 0) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def same_stance_model_contrasts(meta: pd.DataFrame, binary: pd.DataFrame):
    """Each response minus its same-prompt, exact-stance peer-model mean."""
    cell = ["issue_id", "template_id", "rep", "lang", "stance_num"]
    cell_id = meta[cell].astype(str).agg("|".join, axis=1)
    rows, issue_rows = [], []
    for dimension, construct in binary.columns:
        y = binary[(dimension, construct)].astype(float)
        size = y.groupby(cell_id).transform("size")
        # Compare each model with the mean of the OTHER models in the same cell.
        peer_mean = (y.groupby(cell_id).transform("sum") - y) / (size - 1)
        valid = size >= 2
        d = meta.loc[valid, ["family", "lang", "model", "origin", "issue_id"]].copy()
        d["contrast"] = (y[valid] - peer_mean[valid]).to_numpy()
        for keys, g in d.groupby(
            ["family", "lang", "model", "origin", "issue_id"], observed=True
        ):
            issue_rows.append({
                "family": keys[0], "lang": keys[1], "model": keys[2],
                "origin": keys[3], "issue_id": keys[4], "dimension": dimension,
                "construct": construct, "n_matched_responses": len(g),
                "mean_vs_same_stance_peers_pp": 100 * g["contrast"].mean(),
            })
        for keys, g in d.groupby(["family", "lang", "model", "origin"], observed=True):
            rows.append({
                "family": keys[0], "lang": keys[1], "model": keys[2],
                "origin": keys[3], "dimension": dimension, "construct": construct,
                "n_matched_responses": len(g),
                "mean_vs_same_stance_peers_pp": 100 * g["contrast"].mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(issue_rows)


def frame_cooccurrence(meta: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    """Descriptive frame bundles within responses (no independence claim)."""
    frame_cols = [c for c in binary.columns if c[0] == "frame"]
    rows = []
    for keys, idx in meta.groupby(["family", "lang", "model", "origin"],
                                  observed=True).indices.items():
        for i, (_, a) in enumerate(frame_cols):
            va = binary.loc[idx, ("frame", a)].to_numpy(bool)
            for _, b in frame_cols[i + 1:]:
                vb = binary.loc[idx, ("frame", b)].to_numpy(bool)
                pa, pb, both = va.mean(), vb.mean(), np.mean(va & vb)
                union = np.mean(va | vb)
                rows.append({
                    "family": keys[0], "lang": keys[1], "model": keys[2],
                    "origin": keys[3], "frame_a": a, "frame_b": b, "n": len(idx),
                    "share_a": pa, "share_b": pb, "share_both": both,
                    "excess_cooccurrence_pp": 100 * (both - pa * pb),
                    "jaccard": both / union if union else np.nan,
                })
    return pd.DataFrame(rows)


def same_stance_language_contrasts(meta: pd.DataFrame,
                                   binary: pd.DataFrame) -> pd.DataFrame:
    """ZH-minus-EN difference where model, prompt content, and stance all match."""
    keys = ["rep", "model", "issue_id", "template_id", "stance_num"]
    base = meta[keys + ["family", "origin", "lang"]].copy()
    rows = []
    for dimension, construct in binary.columns:
        d = base.copy()
        d["y"] = binary[(dimension, construct)].to_numpy()
        wide = d.pivot_table(index=keys + ["family", "origin"], columns="lang",
                             values="y", aggfunc="first").dropna(subset=["en", "zh"])
        wide["difference"] = wide["zh"] - wide["en"]
        wide = wide.reset_index()
        for gkeys, g in wide.groupby(["family", "model", "origin"], observed=True):
            rows.append({
                "family": gkeys[0], "model": gkeys[1], "origin": gkeys[2],
                "dimension": dimension, "construct": construct,
                "n_exact_stance_language_pairs": len(g),
                "en_share": g["en"].mean(), "zh_share": g["zh"].mean(),
                "zh_minus_en_pp": 100 * g["difference"].mean(),
            })
    return pd.DataFrame(rows)


def validation_sample(meta: pd.DataFrame, count: pd.DataFrame, fore: pd.DataFrame,
                      lead: pd.DataFrame,
                      components_by_lang: dict, regexes_by_lang: dict) -> pd.DataFrame:
    """Two deterministic hits and one non-hit per construct and language."""
    rng = np.random.default_rng(SEED)
    records = []
    for lang in ["en", "zh"]:
        lang_idx = meta.index[
            (meta["lang"] == lang)
            & (~meta["issue_id"].isin(VALIDATION_EXCLUDE_ISSUES))
        ]
        for dimension, construct in count.columns:
            key = (dimension, construct)
            values = count.loc[lang_idx, key]
            front = fore.loc[lang_idx, key]
            hit_front = values.index[(values > 0) & (front > 0)].to_numpy()
            hit_body = values.index[(values > 0) & (front == 0)].to_numpy()
            nonhit = values.index[values == 0].to_numpy()
            picks = []
            if len(hit_front):
                picks.append((int(rng.choice(hit_front)), "hit_foreground"))
            if len(hit_body):
                picks.append((int(rng.choice(hit_body)), "hit_later"))
            elif len(values.index[values > 0]):
                picks.append((int(rng.choice(values.index[values > 0])), "hit_anywhere"))
            if len(nonhit):
                picks.append((int(rng.choice(nonhit)), "nonhit"))

            rxs = regexes_by_lang[lang][key]
            for idx, sample_type in picks:
                row = meta.loc[idx]
                text = str(row["response"])
                matched = []
                for label, rx in zip(components_by_lang[lang][key], rxs):
                    if rx.search(text):
                        matched.append(label)
                detected = [
                    other_construct
                    for other_dimension, other_construct in count.columns
                    if other_dimension == dimension and count.loc[idx, (
                        other_dimension, other_construct)] > 0
                ]
                records.append({
                    "sample_id": f"V{len(records)+1:04d}", "language": lang,
                    "dimension": dimension, "construct": construct,
                    "sample_type": sample_type,
                    "automatic_match": "yes" if sample_type != "nonhit" else "no",
                    "automatic_opening": (
                        "yes" if fore.loc[idx, key] > 0 else "no"),
                    "automatic_paragraph_lead": (
                        "yes" if lead.loc[idx, key] > 0 else "no"),
                    "matched_components": "; ".join(matched),
                    "all_detected_same_dimension": "; ".join(detected),
                    "issue_id": row["issue_id"], "family": row["family"],
                    "model": row["model"], "origin": row["origin"],
                    "stance": int(row["stance_num"]), "template_id": row["template_id"],
                    "rep": row["rep"], "excerpt": first_excerpt(text, rxs),
                    "full_response": text,
                    "review_status": "Needs review", "construct_present": "",
                    "foregrounded": "", "paragraph_lead_salience": "",
                    "endorsed_or_used": "",
                    "false_positive_or_missed_phrase": "", "review_notes": "",
                })
    return pd.DataFrame(records)


def codebook_rows() -> pd.DataFrame:
    rows = []
    for construct, spec in FRAME_COMPONENTS.items():
        rows.append({
            "dimension": "frame", "construct": construct,
            "definition": FRAME_DEFINITIONS[construct],
            "english_components": "; ".join(spec["en"]),
            "chinese_components": "; ".join(spec["zh"]),
            "review_decision": "Pending", "review_notes": "",
        })
    dimension_definition = {
        "actor": ("Actor SALIENCE: the actor is named/made prominent in the response. "
                  "Code present when the actor is mentioned; this is NOT a judgment of "
                  "causal agency (who acts vs. who is acted upon) -- agency is coded only "
                  "in the manual close reading."),
        "mechanism": "Governance intervention proposed or discussed. Direct bilingual pattern.",
        "risk_type": "Concrete harm invoked to justify a position. Direct bilingual pattern.",
    }
    for dimension, source in [
        ("actor", ACTOR_PATTERNS), ("mechanism", MECHANISM_PATTERNS),
        ("risk_type", RISK_PATTERNS),
    ]:
        for construct, spec in source.items():
            rows.append({
                "dimension": dimension, "construct": construct,
                "definition": dimension_definition[dimension],
                "english_components": spec["en"], "chinese_components": spec["zh"],
                "review_decision": "Pending", "review_notes": "",
            })
    return pd.DataFrame(rows)


def run_stance_conditioned_framing() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    meta = load_analysis_rows()
    print(f"{len(meta):,} non-refusal responses with stance", flush=True)

    counts, foregrounds, paragraph_leads = [], [], []
    components_by_lang, regexes_by_lang = {}, {}
    for lang in ["en", "zh"]:
        idx = meta.index[meta["lang"] == lang]
        print(f"Matching {lang.upper()} constructs ({len(idx):,} responses)...", flush=True)
        c, f, lead, components, regexes = match_constructs(
            meta.loc[idx, "response"], lang)
        c.index, f.index, lead.index = idx, idx, idx
        counts.append(c)
        foregrounds.append(f)
        paragraph_leads.append(lead)
        components_by_lang[lang], regexes_by_lang[lang] = components, regexes
    count = pd.concat(counts).sort_index()
    fore = pd.concat(foregrounds).sort_index()
    lead = pd.concat(paragraph_leads).sort_index()
    binary = (count > 0).astype(np.int8)

    # Persist the response-level binary matrix once so follow-up summaries do not need
    # to rescan 108k long responses.  Column names encode measure, dimension, construct.
    cache_meta = meta[[
        "rep", "model", "origin", "prompt_id", "issue_id", "family", "template_id",
        "lang", "stance_num",
    ]].copy()
    cache_features = {}
    for dimension, construct in binary.columns:
        cache_features[f"mention::{dimension}::{construct}"] = binary[
            (dimension, construct)].to_numpy(np.int8)
        cache_features[f"opening::{dimension}::{construct}"] = (fore[
            (dimension, construct)] > 0).to_numpy(np.int8)
        cache_features[f"paragraph_lead::{dimension}::{construct}"] = (lead[
            (dimension, construct)] > 0).to_numpy(np.int8)
    cache = pd.concat(
        [cache_meta.reset_index(drop=True), pd.DataFrame(cache_features)], axis=1)
    cache.to_parquet(TAB / "nb5i_response_constructs.parquet", index=False)

    prevalence = prevalence_table(meta, count, fore, lead)
    model_contrasts, issue_contrasts = same_stance_model_contrasts(meta, binary)
    language_contrasts = same_stance_language_contrasts(meta, binary)
    cooccurrence = frame_cooccurrence(meta, binary)
    review = validation_sample(
        meta, count, fore, lead, components_by_lang, regexes_by_lang)
    codebook = codebook_rows()

    prevalence.to_csv(TAB / "nb5i_frame_prevalence.csv", index=False)
    model_contrasts.to_csv(TAB / "nb5i_same_stance_model_contrasts.csv", index=False)
    issue_contrasts.to_csv(
        TAB / "nb5i_same_stance_model_contrasts_by_issue.csv", index=False)
    language_contrasts.to_csv(TAB / "nb5i_same_stance_language_contrasts.csv", index=False)
    cooccurrence.to_csv(TAB / "nb5i_frame_cooccurrence.csv", index=False)
    review.to_csv(REVIEW / "rq5_frame_validation_sample.csv", index=False)
    codebook.to_csv(REVIEW / "rq5_frame_codebook_review.csv", index=False)

    manifest = {
        "seed": SEED, "opening_fraction": OPENING_FRACTION,
        "primary_salience_measure": "paragraph_lead",
        "n_frames": len(FRAME_COMPONENTS),
        "n_analysis_responses": len(meta), "n_validation_rows": len(review),
        "dimensions": codebook.groupby("dimension").size().to_dict(),
        "outputs": [
            str(TAB / "nb5i_frame_prevalence.csv"),
            str(TAB / "nb5i_response_constructs.parquet"),
            str(TAB / "nb5i_same_stance_model_contrasts.csv"),
            str(TAB / "nb5i_same_stance_model_contrasts_by_issue.csv"),
            str(TAB / "nb5i_same_stance_language_contrasts.csv"),
            str(TAB / "nb5i_frame_cooccurrence.csv"),
            str(REVIEW / "rq5_frame_validation_sample.csv"),
            str(REVIEW / "rq5_frame_codebook_review.csv"),
        ],
    }
    (REVIEW / "rq5_frame_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nLargest exact-same-stance model contrasts (B-series frames):")
    x = model_contrasts[
        (model_contrasts["family"] == "B") & (model_contrasts["dimension"] == "frame")
    ].copy()
    x["abs"] = x["mean_vs_same_stance_peers_pp"].abs()
    print(x.nlargest(20, "abs").drop(columns="abs").round(2).to_string(index=False))
    print(f"\nValidation rows: {len(review)} -> {REVIEW}/rq5_frame_validation_sample.csv")




# --- nb5j_length_adjusted_frame_contrasts.py ---


import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "evaluation/analysis")
from analysis_helpers import load_responses  # noqa: E402


TAB = Path("evaluation/analysis/outputs/tables")
CACHE = TAB / "nb5i_response_constructs.parquet"


def run_length_adjusted_frame_contrasts() -> None:
    d = pd.read_parquet(CACHE)
    lengths = load_responses()[["rep", "model", "prompt_id", "response"]].copy()
    lengths["response_chars"] = lengths["response"].astype(str).str.len().clip(lower=1)
    lengths = lengths.drop(columns="response")
    d = d.merge(lengths, on=["rep", "model", "prompt_id"], how="left",
                validate="one_to_one")
    if d["response_chars"].isna().any():
        raise AssertionError("Some cached construct rows lack a response-length match")
    d["log_chars"] = np.log(d["response_chars"])

    mention_cols = [c for c in d if c.startswith("mention::")]
    rows, issue_rows = [], []
    for family in sorted(d["family"].unique()):
        for lang in ["en", "zh"]:
            base = d[(d["family"] == family) & (d["lang"] == lang)].copy()
            cell_cols = ["issue_id", "template_id", "rep", "stance_num"]
            cell = base[cell_cols].astype(str).agg("|".join, axis=1)
            size = cell.groupby(cell).transform("size")
            valid = size >= 2
            base, cell, size = base.loc[valid].copy(), cell[valid], size[valid]
            loglen = base["log_chars"]
            peer_loglen = (loglen.groupby(cell).transform("sum") - loglen) / (size - 1)
            within_loglen = loglen - loglen.groupby(cell).transform("mean")

            for col in mention_cols:
                _, dimension, construct = col.split("::", 2)
                y = base[col].astype(float)
                peer_y = (y.groupby(cell).transform("sum") - y) / (size - 1)
                within_y = y - y.groupby(cell).transform("mean")
                denom = float(np.dot(within_loglen, within_loglen))
                beta = float(np.dot(within_loglen, within_y) / denom) if denom else 0.0
                raw = y - peer_y
                adjusted = raw - beta * (loglen - peer_loglen)

                work = base[["model", "origin", "issue_id"]].copy()
                work["raw"] = raw.to_numpy()
                work["adjusted"] = adjusted.to_numpy()
                work["log_length_gap"] = (loglen - peer_loglen).to_numpy()
                for keys, g in work.groupby(["model", "origin"], observed=True):
                    rows.append({
                        "family": family, "lang": lang, "model": keys[0],
                        "origin": keys[1], "dimension": dimension,
                        "construct": construct, "n_matched_responses": len(g),
                        "length_slope_within_cell": beta,
                        "raw_contrast_pp": 100 * g["raw"].mean(),
                        "length_adjusted_contrast_pp": 100 * g["adjusted"].mean(),
                        "mean_log_length_gap": g["log_length_gap"].mean(),
                    })
                for keys, g in work.groupby(["model", "origin", "issue_id"], observed=True):
                    issue_rows.append({
                        "family": family, "lang": lang, "model": keys[0],
                        "origin": keys[1], "issue_id": keys[2], "dimension": dimension,
                        "construct": construct, "n_matched_responses": len(g),
                        "length_slope_within_cell": beta,
                        "raw_contrast_pp": 100 * g["raw"].mean(),
                        "length_adjusted_contrast_pp": 100 * g["adjusted"].mean(),
                    })

    out, by_issue = pd.DataFrame(rows), pd.DataFrame(issue_rows)
    out.to_csv(TAB / "nb5j_length_adjusted_model_contrasts.csv", index=False)
    by_issue.to_csv(TAB / "nb5j_length_adjusted_model_contrasts_by_issue.csv", index=False)

    # Compact frame-only diagnostic table for the human codebook review.
    prevalence = pd.read_csv(TAB / "nb5i_frame_prevalence.csv")
    p = prevalence[(prevalence["family"] == "B") &
                   (prevalence["dimension"] == "frame")].copy()
    pooled = []
    for (lang, construct), g in p.groupby(["lang", "construct"], observed=True):
        pooled.append({
            "lang": lang, "construct": construct,
            "mention_share_B": np.average(g["mention_share"], weights=g["n"]),
            "foreground_share_B": np.average(g["foreground_share"], weights=g["n"]),
        })
    pooled = pd.DataFrame(pooled)
    mention = pooled.pivot(index="construct", columns="lang", values="mention_share_B")
    foreground = pooled.pivot(index="construct", columns="lang", values="foreground_share_B")

    adj = out[(out["family"] == "B") & (out["dimension"] == "frame")].copy()
    strongest = []
    for (lang, construct), g in adj.groupby(["lang", "construct"], observed=True):
        row = g.loc[g["length_adjusted_contrast_pp"].abs().idxmax()]
        strongest.append({
            "lang": lang, "construct": construct,
            "strongest_adjusted_model": row["model"],
            "strongest_adjusted_contrast_pp": row["length_adjusted_contrast_pp"],
        })
    strongest = pd.DataFrame(strongest)
    strong_model = strongest.pivot(index="construct", columns="lang",
                                   values="strongest_adjusted_model")
    strong_value = strongest.pivot(index="construct", columns="lang",
                                   values="strongest_adjusted_contrast_pp")

    diag = pd.DataFrame(index=mention.index)
    diag["en_mention_share_B"] = mention["en"]
    diag["zh_mention_share_B"] = mention["zh"]
    diag["en_foreground_share_B"] = foreground["en"]
    diag["zh_foreground_share_B"] = foreground["zh"]
    diag["en_strongest_adjusted_model"] = strong_model["en"]
    diag["en_strongest_adjusted_contrast_pp"] = strong_value["en"]
    diag["zh_strongest_adjusted_model"] = strong_model["zh"]
    diag["zh_strongest_adjusted_contrast_pp"] = strong_value["zh"]
    diag["review_flag"] = np.where(
        (diag["en_mention_share_B"] > 0.80) | (diag["zh_mention_share_B"] > 0.80),
        "Presence saturates; prioritize foregrounding", "",
    )
    diag.reset_index().to_csv(
        "evaluation/analysis/outputs/manual_coding/rq5_frame_review_diagnostics.csv",
        index=False,
    )

    x = out[(out["family"] == "B") & (out["dimension"] == "frame")].copy()
    x["change"] = x["length_adjusted_contrast_pp"] - x["raw_contrast_pp"]
    x["abs_raw"] = x["raw_contrast_pp"].abs()
    print("Largest B-series frame contrasts before/after length adjustment:")
    print(x.nlargest(24, "abs_raw").drop(columns="abs_raw").round(2).to_string(index=False))




# --- nb5j_per_issue_framing.py ---

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "evaluation/analysis")
sys.path.insert(0, "evaluation/scripts/06_analysis")
from analysis_helpers import load_master  # noqa: E402
from viz_style import INK, SEQUENTIAL, apply_style, save  # noqa: E402

FIG = Path("evaluation/analysis/outputs/figures")
TAB = Path("evaluation/analysis/outputs/tables")
FRAMES = list(FRAME_COMPONENTS)  # fixed order


def issue_order(meta, fam):
    return sorted(meta[meta.family == fam].issue_id.unique(), key=lambda x: int(x[1:]))


def frame_presence(meta):
    """Attach one binary column per frame (present anywhere in the response)."""
    parts = []
    for lang in ["en", "zh"]:
        idx = meta.index[meta.lang == lang]
        c, *_ = match_constructs(meta.loc[idx, "response"], lang)
        c.index = idx
        parts.append((c.xs("frame", axis=1, level="dimension") > 0).astype(int))
    return meta.join(pd.concat(parts))


def shift_matrix(meta, issues):
    """frames x issues matrix of (ZH share - EN share) in percentage points."""
    M = np.full((len(FRAMES), len(issues)), np.nan)
    en_M, zh_M = M.copy(), M.copy()
    for j, iss in enumerate(issues):
        d = meta[meta.issue_id == iss]
        en = d[d.lang == "en"][FRAMES].mean() * 100
        zh = d[d.lang == "zh"][FRAMES].mean() * 100
        for i, fr in enumerate(FRAMES):
            en_M[i, j], zh_M[i, j] = en[fr], zh[fr]
    return zh_M - en_M, en_M, zh_M


def fig_shift(shift, issues, labels, frames):
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#c33534", "#ef9d9c", "#f4f4f2", "#86b6ef", "#1c5cab"])
    lim = np.nanmax(np.abs(shift))
    fig, ax = plt.subplots(figsize=(0.62 * len(issues) + 3.2, 0.42 * len(frames) + 1.4))
    im = ax.imshow(shift, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(issues)), labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(frames)), frames, fontsize=8)
    for i in range(len(frames)):
        for j in range(len(issues)):
            v = shift[i, j]
            if np.isfinite(v) and abs(v) >= 8:
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_title("Prompting in Chinese shifts the frame: ZH − EN frame prevalence "
                 "(pp), by issue", fontsize=10, fontweight="bold")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="ZH − EN share (pp);  blue = more in Chinese")
    save(fig, FIG / "nb5j_frame_shift_by_issue")


def fig_prevalence(en_M, zh_M, issues, labels, frames):
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, axes = plt.subplots(1, 2, figsize=(1.1 * len(issues) + 3.5, 0.4 * len(frames) + 1.6))
    for ax, M, lang in zip(axes, [en_M, zh_M], ["EN", "ZH"]):
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(issues)), labels, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(frames)), frames if lang == "EN" else [], fontsize=7.5)
        ax.set_title(lang, fontweight="bold")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Frame prevalence by issue (% of responses)", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.7, label="% responses")
    save(fig, FIG / "nb5j_frame_prevalence_by_issue")


def run_per_issue_framing():
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    meta = frame_presence(load_analysis_rows())

    m = load_master()
    cat = m.drop_duplicates("issue_id").set_index("issue_id")["issue_category"].to_dict()

    # full per-issue table, all families
    rows = []
    for iss in sorted(meta.issue_id.unique()):
        d = meta[meta.issue_id == iss]
        en = d[d.lang == "en"][FRAMES].mean() * 100
        zh = d[d.lang == "zh"][FRAMES].mean() * 100
        for fr in FRAMES:
            rows.append({"issue_id": iss, "family": d.family.iloc[0],
                         "category": cat.get(iss, ""), "frame": fr,
                         "en_share": round(en[fr], 1), "zh_share": round(zh[fr], 1),
                         "zh_minus_en": round(zh[fr] - en[fr], 1)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(TAB / "nb5j_per_issue_frame_share.csv", index=False)

    # headline: B-series shift + prevalence figures
    issues = issue_order(meta, "B")
    labels = [f"{i}\n{cat.get(i,'')[:12]}" for i in issues]
    shift, en_M, zh_M = shift_matrix(meta, issues)
    # order frame rows by mean ZH−EN (most-Chinese frames on top, most-English at bottom)
    ordr = np.argsort(-np.nanmean(shift, axis=1))
    frames_ord = [FRAMES[i] for i in ordr]
    fig_shift(shift[ordr], issues, labels, frames_ord)
    fig_prevalence(en_M[ordr], zh_M[ordr], issues, labels, frames_ord)

    # console: which frames shift most consistently across issues (mean |shift| sign)
    sh = pd.DataFrame(shift, index=FRAMES, columns=issues)
    consistency = pd.DataFrame({
        "mean_zh_minus_en_pp": sh.mean(axis=1).round(1),
        "n_issues_zh_higher": (sh > 0).sum(axis=1),
        "n_issues_en_higher": (sh < 0).sum(axis=1),
    }).sort_values("mean_zh_minus_en_pp")
    print("Frame language-shift consistency across the 11 B issues "
          "(mean ZH−EN pp; +ve = Chinese foregrounds it more):\n")
    print(consistency.to_string())
    print(f"\ntable -> {TAB}/nb5j_per_issue_frame_share.csv")
    print(f"figs  -> {FIG}/nb5j_frame_shift_by_issue.pdf, nb5j_frame_prevalence_by_issue.pdf")




# --- nb5n_frame_contrast_heatmap.py ---

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, "evaluation/analysis")
from viz_style import (  # noqa: E402
    INK, INK_MUTED, MODEL_ORDER, STANCE_COLOR, apply_style, save,
)

TAB = Path("evaluation/analysis/outputs/tables")
FIG = Path("evaluation/analysis/outputs/figures")
CONTRASTS = TAB / "nb5j_length_adjusted_model_contrasts.csv"

FAMILY = "B"  # the contested issues: the headline set
LANGS = [("en", "English prompts"), ("zh", "Chinese prompts")]
MODEL_DISPLAY = {"claude": "Claude", "gpt4o": "GPT-4o", "llama": "Llama",
                 "deepseek": "DeepSeek", "qwen": "Qwen", "mistral": "Mistral"}
# MODEL_ORDER is US, US, US, CN, CN, EU -- origin blocks are already contiguous
BLOCK_EDGES = [3, 5]
ORIGIN_BANDS = [("US", 0, 3), ("China", 3, 5), ("EU", 5, 6)]

# shorter row labels; the dictionary's own names are too long for an axis
ROW_LABEL = {
    "privacy/consumer protection": "privacy / consumer protection",
    "innovation/economic dynamism": "innovation / economic dynamism",
    "geopolitical competition/restriction": "geopolitical competition",
    "catastrophic/existential risk": "catastrophic / existential risk",
    "fundamental rights/dignity": "fundamental rights / dignity",
    "national security": "national security",
    "everyday safety/reliability": "everyday safety / reliability",
    "transparency/accountability": "transparency / accountability",
    "multilateral cooperation": "multilateral cooperation",
    "openness/access/inclusion": "openness / access / inclusion",
    "precautionary/binding oversight": "precautionary oversight",
    "sovereignty/state capacity": "sovereignty / state capacity",
    "regulatory flexibility/voluntarism": "regulatory flexibility",
    "competitiveness/tech leadership": "competitiveness / tech leadership",
    "social stability/collective good": "social stability / collective good",
}


def matrices():
    d = pd.read_csv(CONTRASTS)
    d = d[(d.dimension == "frame") & (d.family == FAMILY)]
    out = {}
    for lang, _ in LANGS:
        piv = d[d.lang == lang].pivot(index="construct", columns="model",
                                      values="length_adjusted_contrast_pp")
        out[lang] = piv[MODEL_ORDER]
    return out


def draw(mats, rows, out_name, figsize, fontsize):
    apply_style()
    # diverging, white at zero -- same pair as the RQ3 shift heatmap, so a reader who
    # has seen that figure reads this one the same way
    cmap = LinearSegmentedColormap.from_list(
        "contrast", [STANCE_COLOR[5], "#ffffff", STANCE_COLOR[1]])
    vmax = 25.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (lang, title) in zip(axes, LANGS):
        M = mats[lang].reindex(rows)[MODEL_ORDER].to_numpy()
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=fontsize,
                        color="white" if abs(v) > 0.62 * vmax else INK, zorder=3)
        ax.set_xticks(range(len(MODEL_ORDER)),
                      [MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=8, rotation=90)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=26)
        # white hairlines between cells, as in the RQ3 heatmap
        ax.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        # origin blocks, so the reader can check whether the US columns cohere
        for e in BLOCK_EDGES:
            ax.axvline(e - 0.5, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=4)
        for name, lo, hi in ORIGIN_BANDS:
            ax.text((lo + hi - 1) / 2, -0.95, name, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=INK_MUTED)

    axes[0].set_yticks(range(len(rows)), [ROW_LABEL[r] for r in rows], fontsize=8)

    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.03, ticks=[-vmax, 0, vmax])
    cbar.ax.set_yticklabels(["below\npeers", "same as\npeers", "above\npeers"], fontsize=7)
    cbar.outline.set_visible(False)

    save(fig, FIG / out_name)
    print(f"wrote {out_name}  ({len(rows)} frames)")


def emit_table(n_top=10):
    """Option C: the n largest model-vs-peer gaps as a booktabs tabular."""
    d = pd.read_csv(CONTRASTS)
    d = d[(d.dimension == "frame") & (d.family == FAMILY)].copy()
    d["abs"] = d["length_adjusted_contrast_pp"].abs()
    top = d.sort_values("abs", ascending=False).head(n_top)

    lines = [r"\begin{tabular}{@{}lllr@{}}", r"\toprule",
             r"Model & Frame & Prompts & Gap (pp) \\", r"\midrule"]
    for _, r in top.iterrows():
        lines.append(f"{MODEL_DISPLAY[r.model]} & {ROW_LABEL[r.construct]} & "
                     f"{'English' if r.lang == 'en' else 'Chinese'} & "
                     f"${r.length_adjusted_contrast_pp:+.1f}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_rq5_frames.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {TAB / 'tab_rq5_frames.tex'}  (top {n_top})")


def run_frame_contrast_heatmap():
    mats = matrices()
    # row order: widest cross-model spread first, pooling the two languages
    spread = pd.concat([(m.max(axis=1) - m.min(axis=1)) for m in mats.values()], axis=1)
    rows_full = spread.max(axis=1).sort_values(ascending=False).index.tolist()

    # the main-text exhibit: the 10 frames with the widest cross-model spread
    draw(mats, rows_full[:10], "nb5n_frame_contrast_heatmap", (8.6, 4.9), 7.4)
    # kept for reference / a possible appendix version
    draw(mats, rows_full, "nb5n_frame_contrast_heatmap_full", (8.6, 6.4), 7.2)
    emit_table()

    print("\nspread (max - min across models), percentage points:")
    print(spread.set_axis(["en", "zh"], axis=1).round(1)
          .sort_values("en", ascending=False).to_string())




if __name__ == "__main__":
    run_keyword_framing_en()
    run_keyword_framing_zh()
    run_stance_conditioned_framing()
    run_length_adjusted_frame_contrasts()
    run_per_issue_framing()
    run_frame_contrast_heatmap()
