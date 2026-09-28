import { expect, test } from '@playwright/test';

// The fake camera (tests/fixtures/camera-sentence.y4m) is one ArabSign test video slowed down
// SLI_TIMESCALE times; the engine gives the sentence model real-time timestamps.
const timescale = process.env.SLI_TIMESCALE ?? '4';

test('sentences mode: a whole signed sentence comes back as text', async ({ page }) => {
  const api: number[] = [];
  page.on('response', (r) => {
    if (r.url().includes('/api/sentence')) api.push(r.status());
  });
  await page.goto(`/?delegate=CPU&debug&timescale=${timescale}#/sign`);
  await page.getByRole('button', { name: 'جمل', exact: true }).click();
  await expect(page.getByRole('button', { name: 'جمل', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'تشغيل الكاميرا' }).click();
  await expect(page.getByRole('button', { name: 'إيقاف الكاميرا' })).toBeVisible({ timeout: 120_000 });

  type Win = { __sliSentences?: { text: string; source: string; final: boolean }[]; __sliFrames: { raw: { pose: unknown; face: unknown; hands: unknown[] } }[] };
  await expect.poll(() => page.evaluate(() => (window as unknown as Win).__sliSentences?.length ?? 0), { timeout: 240_000 }).toBeGreaterThan(0);
  const s = await page.evaluate(() => (window as unknown as Win).__sliSentences![0]);
  console.log('sentence:', s.text, `(${s.source})`, 'api:', api.join(','));
  expect(s.final).toBe(true);
  expect(s.text.length).toBeGreaterThan(0);
  // The sentence models need pose and face as well as hands: most frames where the signer signs
  // must carry them (a mode that stops requesting the body once made the word model useless).
  // Frame counts are not asserted: on a loaded test machine the browser may get ~1 frame a second.
  const [body, all] = await page.evaluate(() => {
    const f = (window as unknown as Win).__sliFrames;
    const signing = f.filter((x) => x.raw.hands.length > 0); // the empty scene before and after has no body
    return [signing.filter((x) => x.raw.pose && x.raw.face).length, signing.length];
  });
  console.log(`signing frames with body and face: ${body}/${all}`);
  expect(body).toBeGreaterThan(0);
  expect(body / all).toBeGreaterThanOrEqual(0.5);
  await expect(page.locator('.sentence-result')).toBeVisible();
});
