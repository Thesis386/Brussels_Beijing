# -*- coding: utf-8 -*-
"""
Chinese framing lexicon for nb5d_zh (RQ5 ZH keyword pass)

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
