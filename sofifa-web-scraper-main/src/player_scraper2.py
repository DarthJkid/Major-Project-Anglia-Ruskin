"""
Modular SoFIFA Player Scraper
Extracts comprehensive player data from sofifa.com
"""
import re
from playwright.async_api import Page


class PlayerScraper:
    """Handles extraction of player data from a single player page"""

    @staticmethod
    def extract_player_id(url: str) -> str:
        """Extract player ID from URL"""
        match = re.search(r'/player/(\d+)/', url)
        return match.group(1) if match else ''

    @staticmethod
    def extract_version(url: str) -> str:
        """Extract version/roster from URL"""
        match = re.search(r'/(\d+)/?$', url.rstrip('/'))
        return match.group(1) if match else ''

    @staticmethod
    def clean_export_text(value):
        """
        Clean text so CSV export stays stable.
        - Remove null bytes
        - Replace line breaks/tabs with spaces
        - Collapse repeated whitespace
        """
        if value is None:
            return ''
        value = str(value)
        value = value.replace('\x00', '')
        value = value.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
        value = re.sub(r'\s+', ' ', value).strip()
        return value

    @staticmethod
    async def scrape_player_data(page: Page, url: str) -> dict:
        """
        Scrape all player data from a player page
        Returns a dictionary with all player attributes
        """
        stats = await page.evaluate("""
            () => {
                const data = {};

                const cleanText = (text) => text ? text.trim() : '';

                const extractNumber = (text) => {
                    if (!text) return '';
                    const match = text.match(/\\d+/);
                    return match ? match[0] : '';
                };

                const parseValue = (text) => {
                    if (!text) return '';
                    text = text.replace(/[€$£,]/g, '');
                    if (text.includes('M')) {
                        return (parseFloat(text) * 1000000).toString();
                    } else if (text.includes('K')) {
                        return (parseFloat(text) * 1000).toString();
                    }
                    return text;
                };

                // Meta description
                const metaDesc = document.querySelector('meta[name="description"]');
                data.description = metaDesc ? metaDesc.content : '';

                // JSON-LD
                const jsonLd = document.querySelector('script[type="application/ld+json"]');
                if (jsonLd) {
                    try {
                        const schema = JSON.parse(jsonLd.textContent);
                        data.full_name = `${schema.givenName || ''} ${schema.familyName || ''}`.trim();
                        data.dob = schema.birthDate || '';
                        data.image = schema.image || '';

                        if (schema.height) {
                            const heightMatch = schema.height.match(/\\d+/);
                            data.height_cm = heightMatch ? heightMatch[0] : '';
                        }
                        if (schema.weight) {
                            const weightMatch = schema.weight.match(/\\d+/);
                            data.weight_kg = weightMatch ? weightMatch[0] : '';
                        }
                    } catch (e) {}
                }

                // Name
                const nameElement = document.querySelector('h1.ellipsis');
                data.name = nameElement ? cleanText(nameElement.textContent) : '';

                if (!data.full_name) {
                    const fullNameElement = document.querySelector('.profile h1');
                    if (fullNameElement) {
                        data.full_name = cleanText(fullNameElement.textContent);
                    }
                }

                // Positions
                const posSpans = document.querySelectorAll('.profile .pos');
                data.positions = Array.from(posSpans).map(span => cleanText(span.textContent)).join(', ');

                // Overall, potential, value, wage
                const gridCols = document.querySelectorAll('.grid .col');
                gridCols.forEach(col => {
                    const sub = col.querySelector('.sub');
                    const em = col.querySelector('em');
                    if (sub && em) {
                        const label = cleanText(sub.textContent).toLowerCase();
                        const value = cleanText(em.textContent);

                        if (label.includes('overall')) {
                            data.overall_rating = extractNumber(value);
                        } else if (label.includes('potential')) {
                            data.potential = extractNumber(value);
                        } else if (label.includes('value')) {
                            data.value = parseValue(value);
                        } else if (label.includes('wage')) {
                            data.wage = parseValue(value);
                        }
                    }
                });

                // Profile / specialities / national team / club
                const profileCols = document.querySelectorAll('.grid.attribute > .col');
                profileCols.forEach(col => {
                    const h5 = col.querySelector('h5');
                    if (!h5) return;

                    const section = cleanText(h5.textContent).toLowerCase();

                    if (section === 'profile') {
                        const labels = col.querySelectorAll('p');
                        labels.forEach(p => {
                            const labelEl = p.querySelector('label');
                            if (!labelEl) return;

                            const labelText = cleanText(labelEl.textContent).toLowerCase();
                            const valueText = cleanText(p.textContent.replace(labelEl.textContent, ''));

                            if (labelText.includes('preferred foot')) {
                                data.preferred_foot = valueText;
                            } else if (labelText.includes('weak foot')) {
                                data.weak_foot = extractNumber(valueText);
                            } else if (labelText.includes('skill moves')) {
                                data.skill_moves = extractNumber(valueText);
                            } else if (labelText.includes('international reputation')) {
                                data.international_reputation = extractNumber(valueText);
                            } else if (labelText.includes('body type')) {
                                data.body_type = valueText;
                            } else if (labelText.includes('real face')) {
                                data.real_face = valueText;
                            } else if (labelText.includes('release clause')) {
                                data.release_clause = parseValue(valueText);
                            }
                        });
                    } else if (section === 'player specialities') {
                        const specialities = Array.from(col.querySelectorAll('a')).map(a => cleanText(a.textContent));
                        data.specialities = specialities.join(', ');
                    } else if (section === 'national team') {
                        const teamLink = col.querySelector('a[href*="/team/"]');
                        if (teamLink) {
                            data.country_name = cleanText(teamLink.textContent);
                            const teamHref = teamLink.getAttribute('href');
                            const teamIdMatch = teamHref ? teamHref.match(/\\/team\\/(\\d+)\\//) : null;
                            data.country_id = teamIdMatch ? teamIdMatch[1] : '';
                        }

                        const leagueLink = col.querySelector('a[href*="/league/"]');
                        if (leagueLink) {
                            data.country_league_name = cleanText(leagueLink.textContent);
                            const leagueHref = leagueLink.getAttribute('href');
                            const leagueIdMatch = leagueHref ? leagueHref.match(/\\/league\\/(\\d+)/) : null;
                            data.country_league_id = leagueIdMatch ? leagueIdMatch[1] : '';
                        }

                        const flagImg = col.querySelector('img.flag');
                        if (flagImg) {
                            const flagSrc = flagImg.getAttribute('data-src') || flagImg.getAttribute('src');
                            data.country_flag = flagSrc || '';
                        }

                        const stars = col.querySelectorAll('svg.star');
                        data.country_rating = stars.length.toString();

                        const posLabels = col.querySelectorAll('p');
                        posLabels.forEach(p => {
                            const labelEl = p.querySelector('label');
                            if (!labelEl) return;

                            const labelText = cleanText(labelEl.textContent).toLowerCase();
                            const valueText = cleanText(p.textContent.replace(labelEl.textContent, ''));

                            if (labelText.includes('position')) {
                                data.country_position = valueText;
                            } else if (labelText.includes('kit number')) {
                                data.country_kit_number = valueText;
                            }
                        });
                    } else if (section === 'club') {
                        const teamLink = col.querySelector('a[href*="/team/"]');
                        if (teamLink) {
                            data.club_name = cleanText(teamLink.textContent);
                            const teamHref = teamLink.getAttribute('href');
                            const teamIdMatch = teamHref ? teamHref.match(/\\/team\\/(\\d+)\\//) : null;
                            data.club_id = teamIdMatch ? teamIdMatch[1] : '';

                            const logoImg = teamLink.querySelector('img.avatar');
                            if (logoImg) {
                                const logoSrc = logoImg.getAttribute('data-src') || logoImg.getAttribute('src');
                                data.club_logo = logoSrc || '';
                            }
                        }

                        const leagueLink = col.querySelector('a[href*="/league/"]');
                        if (leagueLink) {
                            data.club_league_name = cleanText(leagueLink.textContent);
                            const leagueHref = leagueLink.getAttribute('href');
                            const leagueIdMatch = leagueHref ? leagueHref.match(/\\/league\\/(\\d+)/) : null;
                            data.club_league_id = leagueIdMatch ? leagueIdMatch[1] : '';
                        }

                        const stars = col.querySelectorAll('svg.star');
                        data.club_rating = stars.length.toString();

                        const clubLabels = col.querySelectorAll('p');
                        clubLabels.forEach(p => {
                            const labelEl = p.querySelector('label');
                            if (!labelEl) return;

                            const labelText = cleanText(labelEl.textContent).toLowerCase();
                            const valueText = cleanText(p.textContent.replace(labelEl.textContent, ''));

                            if (labelText.includes('position')) {
                                data.club_position = valueText;
                            } else if (labelText.includes('kit number')) {
                                data.club_kit_number = valueText;
                            } else if (labelText.includes('joined')) {
                                data.club_joined = valueText;
                            } else if (labelText.includes('contract')) {
                                data.club_contract_valid_until = valueText;
                            }
                        });
                    }
                });

                // Individual stats
                const statSections = ['attacking', 'skill', 'movement', 'power', 'mentality', 'defending', 'goalkeeping'];

                statSections.forEach(sectionName => {
                    const h5Elements = Array.from(document.querySelectorAll('h5'));
                    const sectionH5 = h5Elements.find(h5 => h5.textContent.trim().toLowerCase() === sectionName);

                    if (sectionH5) {
                        const container = sectionH5.closest('div[class*="col"]') || sectionH5.parentElement;

                        if (container) {
                            const statParagraphs = container.querySelectorAll('p');

                            statParagraphs.forEach(p => {
                                const em = p.querySelector('em');
                                const span = p.querySelector('span[data-tippy-right-start]');

                                if (em && span) {
                                    const statValue = cleanText(em.textContent);
                                    const statName = cleanText(span.textContent);

                                    let normalizedName = statName.toLowerCase()
                                        .replace(/\\s+/g, '_')
                                        .replace(/[^a-z0-9_]/g, '');

                                    if (normalizedName === 'att_position') {
                                        normalizedName = 'att_positioning';
                                    }

                                    const fullStatName = `${sectionName}_${normalizedName}`;
                                    data[fullStatName] = extractNumber(statValue);
                                }
                            });
                        }
                    }
                });

                // PlayStyles
                const playStylesH5 = Array.from(document.querySelectorAll('h5')).find(h5 =>
                    h5.textContent.trim().toLowerCase() === 'playstyles'
                );

                if (playStylesH5) {
                    const container = playStylesH5.closest('div[class*="col"]') || playStylesH5.parentElement;
                    if (container) {
                        const playStyleSpans = container.querySelectorAll('span[data-tippy-right-start]');
                        const playStyles = Array.from(playStyleSpans).map(span => {
                            let text = cleanText(span.textContent);
                            text = text.replace(/\\s*\\+\\+?\\s*$/, '');
                            return text;
                        });
                        data.play_styles = playStyles.join(', ');
                    }
                }

                return data;
            }
        """)

        # Use URL helpers here so both player_id and version are stable
        stats['player_id'] = PlayerScraper.extract_player_id(url)
        stats['version'] = PlayerScraper.extract_version(url)
        stats['url'] = url

        # Normalize keys and clean all values before export
        normalized_stats = {}
        for key, value in stats.items():
            normalized_key = key.lower().replace(' ', '_').replace('-', '_')
            normalized_stats[normalized_key] = PlayerScraper.clean_export_text(value)

        return normalized_stats