/// CLIP zero-shot classification categories.
///
/// Two-level taxonomy: categories (user-facing labels) → subcategories (CLIP prompts).
/// Classification matches against subcategories for precision, then rolls up to category for display.
use std::sync::Mutex;

/// A category with its display name and subcategory prompts.
pub struct TagCategory {
    pub name: &'static str,
    pub icon: &'static str,
    pub subs: &'static [&'static str],
}

/// Result of classifying an image vector against the taxonomy.
#[derive(Debug, Clone)]
pub struct TagResult {
    pub category: &'static str,
    pub icon: &'static str,
    pub subcategory: &'static str,
    pub score: f32,
}

// Minimum similarity threshold to consider a tag relevant
const MIN_SCORE: f32 = 0.26;
const MAX_TAGS: usize = 5;

pub static CATEGORIES: &[TagCategory] = &[
    TagCategory {
        name: "美食",
        icon: "🍽️",
        subs: &[
            "中餐", "西餐", "日料", "韩餐", "泰餐", "烧烤", "火锅", "麻辣烫",
            "小吃", "甜点", "蛋糕", "面包", "饼干", "巧克力", "冰淇淋",
            "面条", "米饭", "粥", "饺子", "包子", "馒头", "煎饼",
            "寿司", "刺身", "拉面", "披萨", "汉堡", "三明治", "沙拉", "牛排",
            "海鲜", "鱼", "虾", "螃蟹", "龙虾", "贝类",
            "烤肉", "炸鸡", "烤鸭", "红烧肉", "糖醋排骨",
            "炒菜", "凉菜", "汤", "炖菜",
            "水果拼盘", "果盘",
            "早餐", "便当", "快餐",
        ],
    },
    TagCategory {
        name: "饮品",
        icon: "🥤",
        subs: &[
            "咖啡", "拿铁", "卡布奇诺", "美式咖啡",
            "茶", "奶茶", "抹茶", "功夫茶",
            "果汁", "冰沙", "柠檬水",
            "啤酒", "红酒", "白酒", "鸡尾酒", "威士忌", "香槟",
            "牛奶", "豆浆", "酸奶",
            "气泡水", "可乐",
        ],
    },
    TagCategory {
        name: "水果",
        icon: "🍎",
        subs: &[
            "苹果", "香蕉", "橙子", "柠檬", "葡萄", "草莓", "蓝莓", "樱桃",
            "桃子", "西瓜", "哈密瓜", "芒果", "菠萝", "猕猴桃", "火龙果",
            "荔枝", "龙眼", "柿子", "石榴", "椰子", "榴莲", "牛油果",
        ],
    },
    TagCategory {
        name: "蔬菜",
        icon: "🥬",
        subs: &[
            "白菜", "西兰花", "胡萝卜", "土豆", "番茄", "黄瓜", "茄子",
            "辣椒", "洋葱", "大蒜", "生姜", "玉米", "豆角", "蘑菇",
            "南瓜", "冬瓜", "丝瓜", "莲藕", "竹笋", "菠菜", "生菜",
        ],
    },
    TagCategory {
        name: "猫",
        icon: "🐱",
        subs: &[
            "猫", "小猫", "猫咪", "橘猫", "黑猫", "白猫", "布偶猫",
            "英短", "美短", "暹罗猫", "波斯猫", "加菲猫",
            "猫咪睡觉", "猫咪玩耍",
        ],
    },
    TagCategory {
        name: "狗",
        icon: "🐕",
        subs: &[
            "狗", "小狗", "狗狗", "金毛", "拉布拉多", "哈士奇", "柴犬",
            "柯基", "泰迪", "博美", "法斗", "边牧", "萨摩耶", "德牧",
            "小型犬", "大型犬",
        ],
    },
    TagCategory {
        name: "动物",
        icon: "🐾",
        subs: &[
            "鸟", "鹦鹉", "麻雀", "老鹰", "天鹅", "鸽子", "孔雀", "火烈鸟",
            "兔子", "仓鼠", "松鼠", "刺猬",
            "乌龟", "蛇", "蜥蜴", "变色龙", "青蛙",
            "马", "牛", "羊", "猪", "鹿", "驴", "骆驼",
            "鸡", "鸭", "鹅",
            "熊猫", "老虎", "狮子", "大象", "长颈鹿", "斑马", "猴子", "猩猩",
            "北极熊", "棕熊", "狐狸", "狼", "豹子",
            "海豚", "鲸鱼", "海豹", "企鹅", "考拉", "袋鼠",
            "蝴蝶", "蜻蜓", "蜜蜂", "瓢虫", "螳螂",
            "金鱼", "锦鲤", "热带鱼", "水母", "珊瑚",
        ],
    },
    TagCategory {
        name: "花卉",
        icon: "🌸",
        subs: &[
            "花", "花束", "花瓶里的花",
            "玫瑰", "向日葵", "樱花", "梅花", "荷花", "莲花",
            "菊花", "兰花", "牡丹", "郁金香", "薰衣草", "桂花",
            "百合", "康乃馨", "茉莉", "桃花", "杜鹃", "紫藤",
            "野花", "油菜花田", "花海", "花园",
        ],
    },
    TagCategory {
        name: "植物",
        icon: "🌿",
        subs: &[
            "树", "大树", "松树", "柳树", "银杏", "枫树", "椰子树", "棕榈树",
            "竹子", "竹林",
            "草地", "草坪", "苔藓",
            "森林", "树林", "丛林", "红树林",
            "盆栽", "绿植", "多肉植物", "仙人掌", "藤蔓",
            "落叶", "枯叶", "树叶",
        ],
    },
    TagCategory {
        name: "山景",
        icon: "🏔️",
        subs: &[
            "山", "高山", "雪山", "山脉", "山峰", "火山",
            "山谷", "峡谷", "悬崖", "岩石",
            "山路", "登山", "徒步",
            "山间云雾", "云海",
        ],
    },
    TagCategory {
        name: "水景",
        icon: "🌊",
        subs: &[
            "大海", "海浪", "海岸", "海滩", "沙滩", "礁石",
            "湖", "湖泊", "湖面倒影",
            "河", "河流", "小溪", "溪流",
            "瀑布", "泉水", "温泉",
            "池塘", "水库", "水坝",
        ],
    },
    TagCategory {
        name: "天空",
        icon: "🌅",
        subs: &[
            "日出", "日落", "夕阳", "朝霞", "晚霞",
            "蓝天", "白云", "蓝天白云", "火烧云",
            "星空", "银河", "月亮", "满月", "星星",
            "极光", "彩虹",
        ],
    },
    TagCategory {
        name: "自然风光",
        icon: "🏞️",
        subs: &[
            "风景", "自然风景", "田园风光",
            "草原", "平原", "田野", "稻田", "麦田", "梯田",
            "沙漠", "戈壁", "绿洲", "沙丘",
            "冰川", "冰山", "冻土",
            "洞穴", "溶洞", "钟乳石",
            "湿地", "沼泽",
        ],
    },
    TagCategory {
        name: "天气",
        icon: "🌤️",
        subs: &[
            "晴天", "多云", "阴天",
            "下雨", "暴雨", "雨后", "雨滴", "水滴",
            "下雪", "雪景", "雪地", "雪花", "霜",
            "大雾", "雾气", "薄雾",
            "闪电", "雷暴", "台风",
        ],
    },
    TagCategory {
        name: "城市",
        icon: "🏙️",
        subs: &[
            "城市", "城市天际线", "都市夜景",
            "高楼大厦", "摩天大楼", "写字楼",
            "街道", "马路", "十字路口", "人行道", "斑马线",
            "小巷", "胡同", "弄堂",
            "广场", "公园", "喷泉",
            "霓虹灯", "灯光", "夜市", "商业街",
            "地标建筑", "地铁站",
        ],
    },
    TagCategory {
        name: "建筑",
        icon: "🏛️",
        subs: &[
            "建筑", "现代建筑", "古建筑", "传统建筑",
            "教堂", "大教堂", "清真寺",
            "寺庙", "佛塔", "神社", "鸟居",
            "城堡", "宫殿", "故宫",
            "桥", "大桥", "石桥", "拱桥",
            "塔", "灯塔", "钟楼",
            "纪念碑", "雕像",
            "亭子", "园林",
        ],
    },
    TagCategory {
        name: "室内",
        icon: "🏠",
        subs: &[
            "客厅", "卧室", "厨房", "浴室", "卫生间",
            "餐厅", "饭厅", "酒吧", "咖啡厅",
            "办公室", "工作台", "书桌",
            "会议室", "教室", "图书馆", "书房",
            "走廊", "楼梯", "阳台", "露台",
            "商场", "超市", "市场",
            "酒店大堂", "酒店房间",
            "健身房", "游泳池",
        ],
    },
    TagCategory {
        name: "家居",
        icon: "🛋️",
        subs: &[
            "沙发", "床", "桌子", "椅子", "凳子",
            "书架", "柜子", "衣柜", "鞋柜",
            "灯", "台灯", "吊灯", "落地灯",
            "窗帘", "地毯", "地板", "瓷砖",
            "花瓶", "相框", "挂画", "镜子", "时钟",
        ],
    },
    TagCategory {
        name: "交通",
        icon: "🚗",
        subs: &[
            "汽车", "轿车", "跑车", "越野车", "卡车", "货车",
            "公交车", "大巴", "出租车",
            "火车", "高铁", "地铁", "有轨电车",
            "飞机", "直升机", "热气球",
            "轮船", "游艇", "帆船", "渡轮", "皮划艇",
            "自行车", "摩托车", "电动车", "滑板车",
            "高速公路", "立交桥", "隧道",
            "停车场", "加油站", "机场", "车站", "港口", "码头",
        ],
    },
    TagCategory {
        name: "人物",
        icon: "👤",
        subs: &[
            "自拍", "肖像", "人像", "证件照",
            "合影", "合照", "团体照",
            "婴儿", "宝宝", "小孩", "儿童", "少年",
            "老人", "长辈",
            "情侣", "新人",
            "背影", "剪影", "侧脸",
            "人群", "路人",
        ],
    },
    TagCategory {
        name: "社交活动",
        icon: "🎉",
        subs: &[
            "婚礼", "婚纱", "婚宴",
            "生日", "生日蛋糕", "生日派对",
            "聚会", "派对", "聚餐", "年夜饭",
            "毕业典礼", "毕业照",
            "颁奖", "典礼", "仪式",
            "野餐", "露营", "烧烤聚会",
        ],
    },
    TagCategory {
        name: "节日",
        icon: "🎊",
        subs: &[
            "春节", "对联", "灯笼", "烟花", "鞭炮", "红包",
            "中秋", "月饼",
            "端午", "粽子", "龙舟",
            "元宵", "汤圆", "花灯",
            "圣诞", "圣诞树", "圣诞老人",
            "万圣", "南瓜灯",
            "情人节", "母亲节", "父亲节",
        ],
    },
    TagCategory {
        name: "运动",
        icon: "⚽",
        subs: &[
            "足球", "篮球", "排球", "网球", "乒乓球", "羽毛球",
            "高尔夫", "棒球", "橄榄球", "冰球",
            "游泳", "跳水", "冲浪", "帆板", "潜水",
            "跑步", "马拉松", "田径", "接力",
            "骑行", "自行车赛",
            "登山", "攀岩", "蹦极",
            "滑雪", "滑冰", "花样滑冰", "滑板",
            "武术", "拳击", "柔道", "跆拳道",
            "瑜伽", "健身", "举重", "体操",
            "赛车", "摩托车赛",
            "台球", "保龄球", "飞镖",
            "钓鱼", "射箭",
        ],
    },
    TagCategory {
        name: "旅行",
        icon: "✈️",
        subs: &[
            "旅行", "旅游", "度假",
            "行李箱", "背包", "护照", "机票", "地图",
            "酒店", "民宿", "帐篷",
            "海边度假", "海岛", "沙滩椅",
            "景区", "景点", "博物馆", "美术馆", "展览",
            "游乐园", "游乐场", "摩天轮", "过山车",
            "动物园", "水族馆",
        ],
    },
    TagCategory {
        name: "艺术",
        icon: "🎨",
        subs: &[
            "绘画", "油画", "水彩画", "国画", "素描", "速写",
            "书法", "毛笔字",
            "雕塑", "石雕", "木雕",
            "涂鸦", "街头艺术", "壁画",
            "陶瓷", "瓷器", "陶艺",
            "手工艺品", "编织", "刺绣",
        ],
    },
    TagCategory {
        name: "音乐",
        icon: "🎵",
        subs: &[
            "吉他", "钢琴", "小提琴", "大提琴", "鼓", "架子鼓",
            "萨克斯", "长笛", "二胡", "古筝", "琵琶",
            "麦克风", "耳机", "音箱",
            "演唱会", "音乐会", "音乐节",
            "乐谱", "唱片", "黑胶唱片",
        ],
    },
    TagCategory {
        name: "文档",
        icon: "📄",
        subs: &[
            "名片", "证件", "身份证", "驾照",
            "收据", "发票", "账单",
            "菜单", "价目表",
            "书", "书页", "书本", "课本",
            "报纸", "杂志", "传单", "海报",
            "信件", "信封", "明信片",
            "便签", "手写笔记", "白板",
            "表格", "图表", "PPT",
        ],
    },
    TagCategory {
        name: "电子产品",
        icon: "📱",
        subs: &[
            "手机", "智能手机",
            "电脑", "笔记本电脑", "台式电脑", "显示器", "屏幕",
            "平板电脑", "iPad",
            "相机", "单反相机", "镜头", "摄像机",
            "键盘", "鼠标", "耳机", "音箱",
            "手表", "智能手表",
            "电视", "投影仪",
            "游戏机", "游戏手柄",
            "无人机", "机器人",
        ],
    },
    TagCategory {
        name: "服饰",
        icon: "👔",
        subs: &[
            "衣服", "T恤", "衬衫", "外套", "夹克", "大衣", "毛衣",
            "裙子", "连衣裙", "短裙",
            "裤子", "牛仔裤", "短裤",
            "运动鞋", "高跟鞋", "靴子", "拖鞋", "凉鞋",
            "帽子", "棒球帽", "草帽", "贝雷帽",
            "包", "手提包", "背包", "钱包",
            "围巾", "领带", "手套",
        ],
    },
    TagCategory {
        name: "饰品",
        icon: "💍",
        subs: &[
            "项链", "手链", "手镯", "戒指", "耳环",
            "手表", "腕表",
            "眼镜", "太阳镜", "墨镜",
            "胸针", "发饰", "头饰",
            "珠宝", "钻石", "宝石",
        ],
    },
    TagCategory {
        name: "玩具",
        icon: "🧸",
        subs: &[
            "玩具", "毛绒玩具", "泰迪熊", "玩偶", "芭比",
            "积木", "乐高",
            "拼图", "魔方",
            "模型", "玩具车", "遥控车",
            "棋盘", "象棋", "围棋", "国际象棋",
            "扑克牌", "桌游",
            "气球",
        ],
    },
    TagCategory {
        name: "厨房",
        icon: "🍳",
        subs: &[
            "厨具", "锅", "平底锅", "砂锅",
            "碗", "盘子", "碟子", "杯子", "玻璃杯",
            "刀", "菜刀", "砧板",
            "筷子", "勺子", "叉子",
            "烤箱", "微波炉", "电饭煲", "榨汁机", "咖啡机",
        ],
    },
    TagCategory {
        name: "医疗健康",
        icon: "🏥",
        subs: &[
            "医院", "诊所", "药房",
            "医生", "护士", "手术室",
            "药品", "药片", "针管", "注射器",
            "口罩", "体温计", "血压计", "听诊器",
            "轮椅", "拐杖", "绷带",
        ],
    },
    TagCategory {
        name: "学习办公",
        icon: "📚",
        subs: &[
            "学校", "大学", "教室", "黑板",
            "笔", "铅笔", "钢笔", "彩笔", "马克笔",
            "笔记本", "日记本", "文件夹",
            "计算器", "尺子", "圆规",
            "办公桌", "文件", "打印机", "复印机",
        ],
    },
    TagCategory {
        name: "宗教文化",
        icon: "🙏",
        subs: &[
            "佛像", "菩萨", "观音", "弥勒",
            "十字架", "耶稳", "圣母",
            "经书", "佛经", "圣经", "古兰经",
            "香炉", "烛台", "祭坛",
            "传统服饰", "汉服", "和服", "旗袍",
        ],
    },
];

/// Cached category embeddings for classification.
struct EmbeddingCache {
    entries: Vec<EmbeddingEntry>,
}

struct EmbeddingEntry {
    category_idx: usize,
    sub_idx: usize,
    vec: Vec<f32>,
}

static CACHE: Mutex<Option<EmbeddingCache>> = Mutex::new(None);

/// Compute cosine similarity between two vectors.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Initialize the embedding cache by encoding all subcategory prompts.
/// Called once on first classification request.
fn ensure_cache(
    embed_fn: &dyn Fn(&str) -> Result<Vec<f32>, String>,
) -> Result<(), String> {
    let mut guard = CACHE.lock().map_err(|e| format!("Cache lock: {e}"))?;
    if guard.is_some() {
        return Ok(());
    }
    let mut entries = Vec::new();
    for (cat_idx, cat) in CATEGORIES.iter().enumerate() {
        for (sub_idx, sub) in cat.subs.iter().enumerate() {
            let prompt = format!("一张{sub}的照片");
            let vec = embed_fn(&prompt)?;
            entries.push(EmbeddingEntry {
                category_idx: cat_idx,
                sub_idx,
                vec,
            });
        }
    }
    tracing::info!(
        "CLIP category embeddings initialized: {} entries across {} categories",
        entries.len(),
        CATEGORIES.len()
    );
    *guard = Some(EmbeddingCache { entries });
    Ok(())
}

/// Classify an image vector against the taxonomy.
/// Returns top matching tags (one per category, sorted by score).
pub fn classify(
    image_vec: &[f32],
    embed_fn: &dyn Fn(&str) -> Result<Vec<f32>, String>,
) -> Result<Vec<TagResult>, String> {
    ensure_cache(embed_fn)?;
    let guard = CACHE.lock().map_err(|e| format!("Cache lock: {e}"))?;
    let cache = guard.as_ref().expect("cache initialized above");

    // Score each subcategory
    let mut best_per_cat: Vec<Option<(usize, f32)>> = vec![None; CATEGORIES.len()];

    for entry in &cache.entries {
        let score = cosine_sim(image_vec, &entry.vec);
        let slot = &mut best_per_cat[entry.category_idx];
        if score >= MIN_SCORE {
            match slot {
                Some((_, prev)) if score > *prev => {
                    *slot = Some((entry.sub_idx, score));
                }
                None => {
                    *slot = Some((entry.sub_idx, score));
                }
                _ => {}
            }
        }
    }

    // Collect results and sort by score
    let mut results: Vec<TagResult> = best_per_cat
        .iter()
        .enumerate()
        .filter_map(|(cat_idx, best)| {
            best.map(|(sub_idx, score)| {
                let cat = &CATEGORIES[cat_idx];
                TagResult {
                    category: cat.name,
                    icon: cat.icon,
                    subcategory: cat.subs[sub_idx],
                    score,
                }
            })
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(MAX_TAGS);
    Ok(results)
}
