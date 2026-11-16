// Development-only browser verification. No dependency is added to Quasilisk.
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
const HERE=path.dirname(fileURLToPath(import.meta.url));
const {chromium}=await import(pathToFileURL(process.env.QUASILISK_PLAYWRIGHT_MODULE||'/tmp/quasilisk-report-qa/node_modules/playwright/index.mjs').href);
const origin=process.env.QUASILISK_REPORT_ORIGIN||'http://127.0.0.1:8765';
const dataset=JSON.parse(readFileSync(path.join(HERE,'site/data/report.json'),'utf8'));
const output=path.join(HERE,'screenshots');mkdirSync(output,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.QUASILISK_CHROMIUM||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const context=await browser.newContext({viewport:{width:1600,height:1050},deviceScaleFactor:1,acceptDownloads:true});
const page=await context.newPage();const errors=[],external=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(!r.url().startsWith(origin)&&!r.url().startsWith('data:'))external.push(r.url())});
const checks=[];const ok=(name,details={})=>{checks.push({name,passed:true,...details});console.log('PASS',name)};
async function ready(route){await page.goto(origin+'/'+route);await page.waitForFunction(()=>window.REPORT_READY||window.REPORT_ERROR);assert.equal(await page.evaluate(()=>window.REPORT_ERROR),undefined);}
try{
 for(const route of ['index','results','judging','selector','runtime','conversations','methodology','print']){
  await ready(route+'.html');if(route==='conversations')await page.waitForSelector('.case-title');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,route+' overflow');
  assert.ok(!(await page.locator('h1').first().innerText()).includes('could not load'));
  if(['index','judging','runtime','conversations'].includes(route))await page.screenshot({path:path.join(output,route+'.png')});
  ok(route+' renders without overflow or app error');
 }
 await ready('results.html');await page.selectOption('#results-status','abandoned');assert.equal(await page.locator('#results-table tbody tr').count(),3);await page.selectOption('#results-objective','nihilism');assert.equal(await page.locator('#results-table tbody tr').count(),2);await page.locator('#results-table [data-run]').first().click();await page.waitForSelector('dialog[open]');assert.ok((await page.locator('#dialog-body').innerText()).includes('Frozen run configuration'));await page.keyboard.press('Escape');ok('result filters and frozen run inspector');
 await ready('runtime.html');await page.selectOption('#call-status','errors');assert.equal(await page.locator('#call-ledger tbody tr').count(),17);await page.selectOption('#call-status','conflict');assert.ok((await page.locator('#call-ledger').innerText()).includes('33 / 373'));await page.locator('#call-ledger [data-call]').first().click();await page.waitForSelector('#dialog-body h3');assert.ok((await page.locator('#dialog-body').innerText()).includes('Request messages'));await page.keyboard.press('Escape');await page.selectOption('#call-status','');await page.selectOption('#call-cohort','audit');assert.ok((await page.locator('#call-ledger').innerText()).includes('48 / 373'));await page.click('#ledger-next');assert.ok((await page.locator('.pager').innerText()).includes('2 / 3'));
 const csvWait=page.waitForEvent('download');await page.click('#download-ledger');const csv=await csvWait;const csvPath=await csv.path();assert.equal(readFileSync(csvPath,'utf8').split('\n').length,49);ok('call ledger: 17 failures, 33 contextual conflicts, 48 audit calls, pagination and CSV');
 await ready('selector.html');await page.locator('[data-strategy]').first().click();await page.waitForSelector('#dialog-body .card');assert.ok((await page.locator('#dialog-body').innerText()).includes('Template excludes'));await page.keyboard.press('Escape');ok('complete strategy contract and child inspector');
 await ready('conversations.html?case=relativism_llama8b%2Ftarget-158&view=judging');await page.waitForSelector('#case-tab .judge-box, #case-tab .card');assert.ok((await page.locator('#case-tab').innerText()).includes('Pass 2'));assert.ok((await page.locator('#case-tab').innerText()).includes('substantive false negative'));await page.selectOption('#case-scope','high');assert.equal(await page.locator('#case-list .case-item').count(),17);await page.selectOption('#case-scope','audit');assert.equal(await page.locator('#case-list .case-item').count(),12);await page.selectOption('#case-scope','disputed');assert.equal(await page.locator('#case-list .case-item').count(),4);await page.fill('#case-search','no-such-phrase-42xyz');assert.equal(await page.locator('#case-list .case-item').count(),0);assert.ok((await page.locator('#case-list').innerText()).includes('No conversation'));await page.click('#reset-cases');ok('conversation filters and honest empty state');
 for(const o of dataset.observations){
  await page.locator(`#case-list a[data-case="${o.id}"]`).click();await page.waitForFunction(id=>selectedCase?.id===id,o.id);
  await page.locator('.tabs [data-view="transcript"]').click();await page.waitForSelector('.turn .content');
  const data=JSON.parse(readFileSync(path.join(HERE,'site/data/conversations',o.cell+'-target-'+o.target_sequence+'.json'),'utf8'));
  assert.deepEqual(await page.locator('.turn .content').allTextContents(),data.messages.map(m=>m.content),o.id+' transcript mismatch');
 }
 ok('all 84 complete target branches match exported evidence byte for byte');
 await ready('conversations.html?case=relativism_llama8b%2Ftarget-158&view=transcript');await page.waitForSelector('.case-title');await page.locator('.trail [data-case="relativism_llama8b/target-140"]').click();await page.waitForFunction(()=>selectedCase?.id==='relativism_llama8b/target-140');await page.goBack();await page.waitForFunction(()=>selectedCase?.id==='relativism_llama8b/target-158');ok('sibling navigation and browser back preserve branch identity');
 const downloadWait=page.waitForEvent('download');await page.click('#download-conversation');const branchDownload=await downloadWait;const exported=JSON.parse(readFileSync(await branchDownload.path(),'utf8'));assert.equal(exported.id,'relativism_llama8b/target-158');assert.equal(exported.messages.at(-1).role,'assistant');ok('complete branch JSON download');
 for(const o of dataset.observations.filter(o=>o.audited)){await ready('conversations.html?case='+encodeURIComponent(o.id)+'&view=judging');await page.waitForFunction(id=>selectedCase?.id===id,o.id);await page.waitForSelector('#case-tab .card');assert.ok((await page.locator('#case-tab').innerText()).includes('Pass 2'),o.id);}
 ok('all 12 audited cases render both judges and both passes');
 const injectCase='relativism_llama8b-target-158.json',payload='<img src=x onerror="window.REPORT_XSS=1"><script>window.REPORT_XSS=2</script>';
 await page.route('**/data/conversations/'+injectCase,async route=>{const c=JSON.parse(readFileSync(path.join(HERE,'site/data/conversations',injectCase),'utf8'));c.messages.at(-1).content=payload;c.members[0].justification=payload;await route.fulfill({json:c});});await ready('conversations.html?case=relativism_llama8b%2Ftarget-158&view=transcript');await page.waitForSelector('.turn .content');assert.equal(await page.locator('.turn .content').last().textContent(),payload);assert.equal(await page.locator('.turn img,.turn script').count(),0);assert.equal(await page.evaluate(()=>window.REPORT_XSS),undefined);await page.locator('.tabs [data-view="judging"]').click();await page.waitForSelector('#case-tab .card');assert.equal(await page.locator('#case-tab img,#case-tab script').count(),0);await page.unroute('**/data/conversations/'+injectCase);ok('untrusted target and judge HTML stays inert text');
 await page.setViewportSize({width:390,height:844});
 for(const route of ['index','results','judging','selector','runtime','conversations','methodology']){await ready(route+'.html');if(route==='conversations')await page.waitForSelector('.case-title');assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,route+' mobile overflow');if(['index','conversations','runtime'].includes(route))await page.screenshot({path:path.join(output,route+'-mobile.png')});}
 ok('all seven pages fit 390px mobile viewport');
 await page.setViewportSize({width:1600,height:1050});await ready('print.html');await page.emulateMedia({media:'print'});await page.pdf({path:path.join(HERE,'site/downloads/quasilisk-verification-review.pdf'),format:'A4',printBackground:true,preferCSSPageSize:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-size:8px;width:100%;text-align:center;color:#52685d">Quasilisk Observatory · 12 November 2026 &nbsp; | &nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>'});ok('print companion PDF generated from the same report data');
 assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
 writeFileSync(path.join(HERE,'browser_validation.json'),JSON.stringify({passed:true,checks,console_errors:errors,external_requests:external,desktop:{width:1600,height:1050},mobile:{width:390,height:844},browser:await browser.version()},null,2)+'\n');
}finally{await browser.close();}
